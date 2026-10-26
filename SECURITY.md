# Security Policy

## Reporting Security Vulnerabilities

We take the security of this project seriously. If you discover a security vulnerability, please do not disclose it publicly through GitHub issues or pull requests.

Instead, please email the maintainers or report a private advisory on GitHub.

## Security Controls and Safeguards

1. **Input Validation and Sanitization**:
   - The FastAPI serving layer validates review text length (max 5,000 characters by default) to protect against denial-of-service (DoS) and memory exhaustion.
   - Batch sizes are capped to prevent unconstrained tensor allocation.
   - Empty or whitespace-only inputs are rejected at the edge with HTTP 422.

2. **Model Deserialization Protection**:
   - Model weights and ONNX graphs are strictly loaded from verified local directory structures.
   - Untrusted pickle loading from external or network sources is strictly avoided.
   - Checksums and schema parity checks are executed prior to runtime initialization.

3. **User Privacy and Data Handling**:
   - Product review inputs are treated as user-generated content.
   - Raw review content is not logged by default.
   - Only non-sensitive request metrics (request ID, token count, latency, response status) are recorded in production logs.

4. **Secrets and Credential Hygiene**:
   - No API tokens, certificates, or private keys are committed to source control.
   - All external endpoints or optional tokenized services require environment variable injection (`.env`).
