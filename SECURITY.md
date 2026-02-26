# Security Policy

## Reporting a Vulnerability

**Do not open public GitHub issues for security vulnerabilities.**

If you discover a security vulnerability in NeuroBrix, please report it responsibly by emailing:

**[security@neurobrix.es](mailto:security@neurobrix.es)**

Include as much of the following as possible:

- Description of the vulnerability
- Steps to reproduce
- Potential impact
- Suggested fix (if any)

### Response Timeline

| Stage | Timeframe |
|-------|-----------|
| Acknowledgment | Within 48 hours |
| Initial assessment | Within 5 business days |
| Fix development | Depends on severity |
| Public disclosure | After fix is released |

We will work with you to understand the issue and coordinate disclosure. We appreciate your help in keeping NeuroBrix and its users safe.

## Scope

This security policy covers:

- The NeuroBrix runtime engine (`neurobrix` Python package)
- The `.nbx` container format parser and validator
- The NeuroBrix CLI tools
- The serving daemon and its network-facing components
- The NeuroBrix Hub API at `neurobrix.es`

### Out of Scope

- Third-party dependencies (report to their respective maintainers)
- Model weights or training data (not controlled by NeuroBrix)
- Social engineering attacks

## Supported Versions

| Version | Supported |
|---------|-----------|
| Latest release | Yes |
| Previous minor | Security fixes only |
| Older | No |

We recommend always running the latest version of NeuroBrix.

## Security Considerations

### NBX Container Security

The `.nbx` container format executes computation graphs. While NeuroBrix validates container integrity, users should:

- Only load `.nbx` files from trusted sources (NeuroBrix Hub or self-traced models)
- Use `neurobrix validate --level deep --strict` before running untrusted containers
- Be aware that containers can contain arbitrary computation graphs

### Serving Daemon

The `neurobrix serve` daemon binds to `localhost` by default. If exposed to a network:

- Use a reverse proxy with authentication
- Do not expose the daemon directly to the internet
- Monitor resource usage and set appropriate timeouts

### Dependencies

NeuroBrix depends on PyTorch, safetensors, and other packages. Keep all dependencies updated:

```bash
pip install --upgrade neurobrix
```

## Credit

We gratefully acknowledge security researchers who report vulnerabilities responsibly. With your permission, we will credit you in the release notes.

---

NeuroBrix is developed and maintained by **WizWorks OÜ**, a property of **Neural Networks Holding LTD**.
