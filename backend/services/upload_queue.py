"""Upload Queue Service for managing batch paper processing."""

import asyncio
import logging
import uuid
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set

from sqlalchemy import create_engine, select, update
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker
from sqlalchemy.orm import Session

from config import Settings
from database.models import UploadTask
from services.paper_library import PaperLibraryService

logger = logging.getLogger(__name__)


class UploadQueueService:
    """Manages the upload processing queue.

    Features:
    - Database-backed task persistence
    - Priority queue (higher priority processed first)
    - Concurrent processing with configurable workers (default: 2)
    - Progress callbacks for SSE streaming
    - Graceful shutdown support
    """

    def __init__(
        self,
        session_maker: async_sessionmaker[AsyncSession],
        paper_library: PaperLibraryService,
        max_workers: int = 2,
    ):
        """Initialize the upload queue service.

        Args:
            session_maker: Async session factory for database access
            paper_library: PaperLibraryService instance for indexing papers
            max_workers: Maximum number of concurrent processing workers
        """
        self.session_maker = session_maker
        self.paper_library = paper_library
        self.max_workers = max_workers
        self._executor = ThreadPoolExecutor(max_workers=max_workers)
        self._progress_callbacks: Dict[str, List[Callable]] = {}  # batch_id -> callbacks
        self._running = False
        self._processing_tasks: Set[str] = set()
        self._process_task: Optional[asyncio.Task] = None

        # Create a synchronous engine for thread-pool progress updates
        settings = Settings()
        self._sync_engine = create_engine(settings.database_url, echo=False)

    async def start(self) -> None:
        """Start the queue processor."""
        if self._running:
            return

        self._running = True

        # Recover any interrupted tasks from previous runs
        await self._recover_interrupted_tasks()

        # Start the processing loop
        self._process_task = asyncio.create_task(self._process_loop())
        logger.info("Upload queue service started")

    async def stop(self) -> None:
        """Stop the queue processor gracefully."""
        self._running = False

        if self._process_task:
            self._process_task.cancel()
            try:
                await self._process_task
            except asyncio.CancelledError:
                pass

        # Wait for in-progress tasks to complete
        while self._processing_tasks:
            await asyncio.sleep(0.5)

        self._executor.shutdown(wait=True)
        self._sync_engine.dispose()
        logger.info("Upload queue service stopped")

    async def _recover_interrupted_tasks(self) -> None:
        """Recover tasks that were interrupted by a server restart."""
        async with self.session_maker() as session:
            # Find tasks that were in processing states
            processing_statuses = ['uploading', 'processing', 'extracting', 'embedding', 'indexing']
            result = await session.execute(
                select(UploadTask).where(UploadTask.status.in_(processing_statuses))
            )
            interrupted_tasks = result.scalars().all()

            for task in interrupted_tasks:
                # Reset to pending if file exists, otherwise mark as error
                if task.file_path and Path(task.file_path).exists():
                    task.status = 'pending'
                    task.current_step = None
                    task.progress_percent = 0
                    task.started_at = None
                    logger.info(f"Recovered interrupted task: {task.id}")
                else:
                    task.status = 'error'
                    task.error_message = 'File not found after server restart'
                    task.completed_at = datetime.utcnow()
                    logger.warning(f"Marked interrupted task as error (file missing): {task.id}")

            await session.commit()

    async def _process_loop(self) -> None:
        """Main processing loop - picks up pending tasks by priority."""
        while self._running:
            try:
                # Check if we have capacity
                if len(self._processing_tasks) >= self.max_workers:
                    await asyncio.sleep(0.5)
                    continue

                # Get next pending task
                task = await self._get_next_task()
                if task:
                    # Process in thread pool (non-blocking)
                    self._processing_tasks.add(task.id)
                    asyncio.create_task(self._process_task_async(task))
                else:
                    # No tasks available, wait before checking again
                    await asyncio.sleep(1.0)

            except Exception as e:
                logger.error(f"Error in process loop: {e}")
                await asyncio.sleep(1.0)

    async def _get_next_task(self) -> Optional[UploadTask]:
        """Get the next pending task, ordered by priority (desc) and created_at (asc)."""
        async with self.session_maker() as session:
            result = await session.execute(
                select(UploadTask)
                .where(UploadTask.status == 'pending')
                .where(UploadTask.file_path.isnot(None))
                .order_by(UploadTask.priority.desc(), UploadTask.created_at.asc())
                .limit(1)
            )
            task = result.scalar_one_or_none()

            if task:
                # Mark as processing to prevent other workers from picking it up
                task.status = 'processing'
                task.started_at = datetime.utcnow()
                await session.commit()

                # Refresh to get updated values
                await session.refresh(task)

            return task

    async def _process_task_async(self, task: UploadTask) -> None:
        """Process a task asynchronously using the thread pool."""
        loop = asyncio.get_event_loop()
        try:
            # Run the synchronous indexing in a thread
            result = await loop.run_in_executor(
                self._executor,
                self._process_task_sync,
                task.id,
                task.batch_id,
                task.file_path,
            )

            # Update task with result
            async with self.session_maker() as session:
                db_task = await session.get(UploadTask, task.id)
                if db_task:
                    if result.get('success'):
                        db_task.status = 'complete'
                        db_task.paper_id = result.get('paper_id')
                        db_task.progress_percent = 100
                        db_task.current_step = 'complete'
                    else:
                        db_task.status = 'error'
                        db_task.error_message = result.get('error', 'Unknown error')
                    db_task.completed_at = datetime.utcnow()
                    await session.commit()

            # Emit completion event
            if result.get('success'):
                self._emit_progress(task.batch_id, {
                    'type': 'task_complete',
                    'taskId': task.id,
                    'paperId': result.get('paper_id'),
                    'chunks': result.get('chunks', 0),
                })
            else:
                self._emit_progress(task.batch_id, {
                    'type': 'task_error',
                    'taskId': task.id,
                    'errorMessage': result.get('error', 'Unknown error'),
                })

            # Check if batch is complete
            await self._check_batch_complete(task.batch_id)

        except Exception as e:
            logger.error(f"Error processing task {task.id}: {e}")
            async with self.session_maker() as session:
                db_task = await session.get(UploadTask, task.id)
                if db_task:
                    db_task.status = 'error'
                    db_task.error_message = str(e)
                    db_task.completed_at = datetime.utcnow()
                    await session.commit()

            self._emit_progress(task.batch_id, {
                'type': 'task_error',
                'taskId': task.id,
                'errorMessage': str(e),
            })
        finally:
            self._processing_tasks.discard(task.id)

    def _process_task_sync(self, task_id: str, batch_id: str, file_path: str) -> Dict[str, Any]:
        """Process a single upload task synchronously (runs in thread pool)."""
        pdf_path = Path(file_path)

        if not pdf_path.exists():
            return {'success': False, 'error': 'File not found'}

        def progress_callback(step: str, data: Dict[str, Any]):
            """Progress callback that emits SSE events with sub-step interpolation."""
            # Map step names to base and next-step percentages
            step_ranges = {
                'processing': (0, 5),
                'extracting': (5, 35),
                'chunking': (35, 45),
                'embedding': (45, 75),
                'indexing': (75, 90),
                'complete': (100, 100),
                'error': (0, 0),
            }

            base, next_base = step_ranges.get(step, (0, 0))
            sub_progress = data.get('sub_progress', 0.0)
            progress_percent = int(base + sub_progress * (next_base - base))

            # Emit progress event
            event = {
                'type': 'task_progress',
                'taskId': task_id,
                'status': step,
                'currentStep': data.get('message', step),
                'progressPercent': progress_percent,
                'paperId': data.get('paper_id'),
            }
            callbacks = self._progress_callbacks.get(batch_id, [])
            logger.info(f"Task {task_id}: {step} {progress_percent}% (callbacks: {len(callbacks)})")
            self._emit_progress(batch_id, event)

            # Update database (synchronously via new connection)
            self._update_task_progress_sync(task_id, step, progress_percent, data.get('message'))

        # Call the paper library indexing method
        result = self.paper_library.index_paper(pdf_path, progress_callback)
        return result

    def _update_task_progress_sync(
        self,
        task_id: str,
        status: str,
        progress_percent: int,
        current_step: Optional[str]
    ) -> None:
        """Update task progress in database synchronously.

        Uses a synchronous engine since this is called from a ThreadPoolExecutor.
        """
        try:
            with Session(self._sync_engine) as session:
                session.execute(
                    update(UploadTask)
                    .where(UploadTask.id == task_id)
                    .values(
                        status=status,
                        progress_percent=progress_percent,
                        current_step=current_step,
                    )
                )
                session.commit()
        except Exception as e:
            logger.error(f"Failed to update task progress in DB: {e}")

    async def _check_batch_complete(self, batch_id: str) -> None:
        """Check if all tasks in a batch are complete and emit batch_complete event."""
        async with self.session_maker() as session:
            result = await session.execute(
                select(UploadTask).where(UploadTask.batch_id == batch_id)
            )
            tasks = result.scalars().all()

            if not tasks:
                return

            completed = sum(1 for t in tasks if t.status == 'complete')
            failed = sum(1 for t in tasks if t.status == 'error')
            total = len(tasks)

            # All tasks are done (either complete or error)
            if completed + failed == total:
                self._emit_progress(batch_id, {
                    'type': 'batch_complete',
                    'batchId': batch_id,
                    'succeeded': completed,
                    'failed': failed,
                    'total': total,
                })

    def _emit_progress(self, batch_id: str, event: Dict[str, Any]) -> None:
        """Emit progress event to all registered callbacks for a batch."""
        callbacks = self._progress_callbacks.get(batch_id, [])
        for callback in callbacks:
            try:
                callback(event)
            except Exception as e:
                logger.error(f"Error in progress callback: {e}")

    def register_progress_callback(self, batch_id: str, callback: Callable[[Dict[str, Any]], None]) -> None:
        """Register a callback for progress updates on a batch."""
        if batch_id not in self._progress_callbacks:
            self._progress_callbacks[batch_id] = []
        self._progress_callbacks[batch_id].append(callback)

    def unregister_progress_callback(self, batch_id: str, callback: Callable[[Dict[str, Any]], None]) -> None:
        """Unregister a progress callback."""
        if batch_id in self._progress_callbacks:
            try:
                self._progress_callbacks[batch_id].remove(callback)
                if not self._progress_callbacks[batch_id]:
                    del self._progress_callbacks[batch_id]
            except ValueError:
                pass

    async def create_batch(self, user_id: Optional[int] = None) -> str:
        """Create a new batch and return its ID."""
        return str(uuid.uuid4())

    async def add_task(
        self,
        batch_id: str,
        filename: str,
        file_size: int,
        user_id: Optional[int] = None,
        priority: int = 0,
    ) -> UploadTask:
        """Add a new task to the queue.

        Args:
            batch_id: The batch this task belongs to
            filename: Original filename
            file_size: Size of the file in bytes
            user_id: Optional user ID
            priority: Task priority (higher = processed first)

        Returns:
            The created UploadTask
        """
        task_id = str(uuid.uuid4())

        async with self.session_maker() as session:
            task = UploadTask(
                id=task_id,
                batch_id=batch_id,
                user_id=user_id,
                filename=filename,
                status='pending',
                progress_percent=0,
                priority=priority,
                file_size=file_size,
            )
            session.add(task)
            await session.commit()
            await session.refresh(task)

            logger.info(f"Created upload task: {task_id} for file: {filename}")
            return task

    async def set_task_file_path(self, task_id: str, file_path: Path) -> None:
        """Set the file path for a task after upload."""
        async with self.session_maker() as session:
            result = await session.execute(
                update(UploadTask)
                .where(UploadTask.id == task_id)
                .values(file_path=str(file_path), status='pending')
            )
            await session.commit()

    async def cancel_task(self, task_id: str) -> bool:
        """Cancel a pending task.

        Returns:
            True if task was cancelled, False if it was already processing/complete
        """
        async with self.session_maker() as session:
            task = await session.get(UploadTask, task_id)
            if not task:
                return False

            if task.status in ['pending', 'uploading']:
                task.status = 'error'
                task.error_message = 'Cancelled by user'
                task.completed_at = datetime.utcnow()
                await session.commit()

                self._emit_progress(task.batch_id, {
                    'type': 'task_error',
                    'taskId': task_id,
                    'errorMessage': 'Cancelled by user',
                })
                return True

            return False

    async def retry_task(self, task_id: str) -> bool:
        """Retry a failed task.

        Returns:
            True if task was queued for retry, False if not in error state
        """
        async with self.session_maker() as session:
            task = await session.get(UploadTask, task_id)
            if not task:
                return False

            if task.status == 'error' and task.file_path and Path(task.file_path).exists():
                task.status = 'pending'
                task.error_message = None
                task.progress_percent = 0
                task.current_step = None
                task.started_at = None
                task.completed_at = None
                await session.commit()

                self._emit_progress(task.batch_id, {
                    'type': 'task_progress',
                    'taskId': task_id,
                    'status': 'pending',
                    'currentStep': 'Queued for retry',
                    'progressPercent': 0,
                })
                return True

            return False

    async def set_task_priority(self, task_id: str, priority: int) -> bool:
        """Set the priority of a pending task.

        Returns:
            True if priority was updated, False if task not found or not pending
        """
        async with self.session_maker() as session:
            task = await session.get(UploadTask, task_id)
            if not task or task.status != 'pending':
                return False

            task.priority = priority
            await session.commit()
            return True

    async def get_batch_status(self, batch_id: str) -> Dict[str, Any]:
        """Get the status of all tasks in a batch."""
        async with self.session_maker() as session:
            result = await session.execute(
                select(UploadTask)
                .where(UploadTask.batch_id == batch_id)
                .order_by(UploadTask.priority.desc(), UploadTask.created_at.asc())
            )
            tasks = result.scalars().all()

            return {
                'batchId': batch_id,
                'tasks': [
                    {
                        'taskId': t.id,
                        'batchId': t.batch_id,
                        'filename': t.filename,
                        'paperId': t.paper_id,
                        'status': t.status,
                        'currentStep': t.current_step,
                        'progressPercent': t.progress_percent,
                        'errorMessage': t.error_message,
                        'priority': t.priority,
                        'fileSize': t.file_size,
                        'createdAt': t.created_at.isoformat() if t.created_at else None,
                    }
                    for t in tasks
                ],
                'total': len(tasks),
                'completed': sum(1 for t in tasks if t.status == 'complete'),
                'failed': sum(1 for t in tasks if t.status == 'error'),
                'inProgress': sum(1 for t in tasks if t.status in ['processing', 'extracting', 'embedding', 'indexing']),
                'pending': sum(1 for t in tasks if t.status == 'pending'),
            }

    async def get_task(self, task_id: str) -> Optional[Dict[str, Any]]:
        """Get a single task by ID."""
        async with self.session_maker() as session:
            task = await session.get(UploadTask, task_id)
            if not task:
                return None

            return {
                'taskId': task.id,
                'batchId': task.batch_id,
                'filename': task.filename,
                'paperId': task.paper_id,
                'status': task.status,
                'currentStep': task.current_step,
                'progressPercent': task.progress_percent,
                'errorMessage': task.error_message,
                'priority': task.priority,
                'fileSize': task.file_size,
                'createdAt': task.created_at.isoformat() if task.created_at else None,
            }
