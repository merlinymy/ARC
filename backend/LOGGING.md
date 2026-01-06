# Backend Logging Documentation

Comprehensive logging system for the Research Paper RAG API with structured logging, request/response tracking, and user operation monitoring.

## Features

### 1. **Multi-Level Logging**
- **Console Logging**: Color-coded output for development
- **File Logging**: Rotating log files for production
- **Access Logging**: Dedicated API request/response logs
- **Error Logging**: Separate error log file

### 2. **Log Files**

All logs are stored in the `logs/` directory (configurable via `LOG_DIR` env var):

```
logs/
├── app.log         # General application logs (INFO+)
├── error.log       # Errors only (ERROR+)
└── access.log      # API requests/responses
```

**Log Rotation**: Automatic rotation when files reach 10MB (app/error) or 20MB (access), keeping 5-10 backups.

### 3. **What Gets Logged**

#### Request Logging
Every API request logs:
- **Request ID**: Unique correlation ID for tracing
- **User ID**: Authenticated user (if logged in)
- **Client IP**: Originating IP address
- **Method & Path**: HTTP method and endpoint
- **Query Parameters**: URL parameters (non-sensitive only)
- **Timestamp**: When request was received

Example:
```
INFO | 2026-01-05 18:30:15 | access:123 | REQUEST | POST /papers/upload/batch/init
  request_id=abc-123, user_id=1, client_ip=192.168.1.100, query_params={}
```

#### Response Logging
Every API response logs:
- **Status Code**: HTTP response status
- **Duration**: Response time in milliseconds
- **Request ID**: Same correlation ID for matching

Example:
```
INFO | 2026-01-05 18:30:16 | access:123 | RESPONSE | POST /papers/upload/batch/init | 200 | 127.45ms
  request_id=abc-123, user_id=1, status_code=200, duration_ms=127.45
```

#### Error Logging
All errors include:
- **Stack trace**: Full exception traceback
- **Error type**: Exception class name
- **Context**: Request details when error occurred
- **User info**: Who encountered the error

Example:
```
ERROR | 2026-01-05 18:31:20 | access:456 | ERROR | GET /papers/abc-123 | PaperNotFoundException
  request_id=abc-123, user_id=1, error_type=PaperNotFoundException
  Traceback:...
```

#### User Operation Logging
Important user actions are logged:
- **Paper upload**: Init batch, file uploads
- **Paper deletion**: Which papers were deleted
- **Search queries**: User search operations
- **Authentication**: Login/logout events

Example:
```
INFO | 2026-01-05 18:35:10 | access:789 | USER_OP | delete_paper | user=1
  request_id=abc-789, user_id=1, operation=delete_paper, details={"paper_id": "paper_123"}
```

## Configuration

### Environment Variables (.env)

```bash
# Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
LOG_LEVEL=INFO

# Directory for log files
LOG_DIR=logs

# Enable file logging (true/false)
ENABLE_FILE_LOGGING=true

# Use JSON format for logs (true/false) - useful for log aggregation
ENABLE_JSON_LOGGING=false

# Enable detailed API access logging (true/false)
ENABLE_ACCESS_LOGGING=true
```

### Log Levels

- **DEBUG**: Verbose output, includes internal details
- **INFO**: General information, normal operations
- **WARNING**: Potential issues, recoverable errors
- **ERROR**: Errors that need attention
- **CRITICAL**: Severe errors, service degradation

### Production Recommendations

```bash
LOG_LEVEL=WARNING          # Reduce noise
ENABLE_FILE_LOGGING=true   # Keep logs on disk
ENABLE_JSON_LOGGING=true   # For log aggregation (ELK, Splunk, etc.)
ENABLE_ACCESS_LOGGING=true # Monitor API usage
```

### Development Recommendations

```bash
LOG_LEVEL=DEBUG            # See everything
ENABLE_FILE_LOGGING=false  # Console only for faster dev
ENABLE_JSON_LOGGING=false  # Human-readable format
ENABLE_ACCESS_LOGGING=true # Debug API issues
```

## Log Format

### Console (Human-Readable)
```
LEVEL | TIMESTAMP | LOGGER:LINE | MESSAGE
  extra_field1=value1, extra_field2=value2
```

### JSON (Machine-Readable)
```json
{
  "timestamp": "2026-01-05T18:30:15.123456",
  "level": "INFO",
  "logger": "access",
  "message": "REQUEST | POST /papers/upload",
  "module": "main",
  "function": "request_middleware",
  "line": 265,
  "request_id": "abc-123",
  "user_id": "1",
  "client_ip": "192.168.1.100"
}
```

## Request Tracing

Every request gets a unique `X-Request-ID` header:
- **Client-provided**: Use existing header if present
- **Auto-generated**: Create UUID if not provided
- **Response header**: Returned in `X-Request-ID` header

Use this ID to trace a request across:
- Application logs
- Access logs
- Error logs
- External systems

Example:
```bash
# Make request with custom ID
curl -H "X-Request-ID: my-trace-123" http://localhost:8000/papers

# Check logs
grep "my-trace-123" logs/access.log
grep "my-trace-123" logs/app.log
```

## Monitoring & Alerts

### Key Metrics to Monitor

1. **Error Rate**: Count of ERROR log entries
2. **Response Time**: `duration_ms` in access logs
3. **Status Codes**: 4xx/5xx response rates
4. **User Operations**: Track upload/delete frequency

### Log Aggregation

For production, integrate with:
- **ELK Stack**: Elasticsearch + Logstash + Kibana
- **Splunk**: Enterprise log management
- **Datadog**: APM + logging
- **CloudWatch**: AWS-native logging

Enable `ENABLE_JSON_LOGGING=true` for easier parsing.

### Sample Queries

**Find all errors for a user:**
```bash
grep "user_id=1" logs/error.log
```

**Find slow requests (>1s):**
```bash
awk -F'|' '/RESPONSE/ {if ($NF ~ /[0-9]+ms/ && $NF+0 > 1000) print}' logs/access.log
```

**Count requests by endpoint:**
```bash
grep "REQUEST" logs/access.log | awk '{print $5,$6}' | sort | uniq -c | sort -rn
```

## Privacy & Security

### Sensitive Data Filtering

The logging system automatically:
- ✅ **Excludes** passwords, tokens, API keys from logs
- ✅ **Redacts** email addresses in query params
- ✅ **Sanitizes** file paths to prevent path traversal info leaks

### What's NOT Logged

- Request/response bodies (to avoid PII/large data)
- Authentication tokens
- Passwords
- API keys
- Session cookies

### What's Logged

- Request paths and methods
- Query parameters (non-sensitive)
- User IDs (for audit trail)
- IP addresses (for security)
- File names (without content)

## Troubleshooting

### Logs not appearing?

1. Check `LOG_LEVEL` - may be set too high
2. Verify `ENABLE_FILE_LOGGING=true`
3. Check permissions on `logs/` directory
4. Restart backend after config changes

### Too many logs?

1. Increase `LOG_LEVEL` to `WARNING` or `ERROR`
2. Disable `ENABLE_ACCESS_LOGGING` if not needed
3. Reduce log retention (edit `logging_config.py`)

### Can't find a specific request?

Use the `X-Request-ID` header to trace:
```bash
# Get request ID from response
curl -i http://localhost:8000/api/health | grep X-Request-ID

# Search logs
grep "request-id-here" logs/*.log
```

## Examples

### Finding Upload Errors

```bash
# All upload errors
grep "upload" logs/error.log

# Failed uploads by user
grep "USER_OP | init_batch_upload" logs/access.log | grep "user_id=1"
```

### Monitoring API Performance

```bash
# Average response time
grep "RESPONSE" logs/access.log | awk -F'|' '{print $NF}' | awk '{sum+=$1; n++} END {print sum/n "ms"}'

# Slowest endpoints
grep "RESPONSE" logs/access.log | sort -t'|' -k7 -rn | head -20
```

### User Activity Audit

```bash
# All operations by user 1
grep "user_id=1" logs/access.log | grep "USER_OP"

# Papers deleted today
grep "delete_paper" logs/access.log | grep "$(date +%Y-%m-%d)"
```

## Development

### Adding Custom Logging

```python
from logging_config import RequestLogger
import logging

logger = logging.getLogger(__name__)

# Simple logging
logger.info("Processing started")
logger.error("Something went wrong", exc_info=True)

# User operation logging
req_logger = RequestLogger(request_id, user_id)
req_logger.log_user_operation("custom_action", {"detail": "value"})
```

### Testing Logging

```python
# Test logging setup
from logging_config import setup_logging

setup_logging(log_level="DEBUG", enable_file=False)
logger.debug("This is a test")
```

## Summary

The logging system provides:
- ✅ Complete audit trail of user operations
- ✅ Request/response tracking with correlation IDs
- ✅ Error monitoring with full context
- ✅ Performance monitoring (response times)
- ✅ Security monitoring (failed auth, rate limits)
- ✅ Production-ready log rotation
- ✅ Privacy-aware (no sensitive data)

Logs are your best friend for debugging, monitoring, and security auditing!
