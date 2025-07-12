# Security Policy

## Supported Versions

| Version | Supported          |
| ------- | ------------------ |
| 1.0.x   | :white_check_mark: |

## Reporting a Vulnerability

If you discover a security vulnerability in this project, please report it by:

1. **Do NOT** create a public GitHub issue
2. Email the maintainers directly
3. Include detailed information about the vulnerability
4. Provide steps to reproduce if possible

We will respond to security reports within 48 hours and work with you to resolve any issues promptly.

## Security Considerations

This application:
- Processes user-uploaded images locally
- Uses AI models that download from Hugging Face
- Does not store personal data permanently
- Implements file type validation
- Limits upload file sizes

## Best Practices

When deploying this application:
- Use HTTPS in production
- Implement rate limiting
- Monitor resource usage
- Keep dependencies updated
- Review uploaded content policies
