# API Reference

This section will contain the API documentation for chat-factory.

## How to Document Your API

1. As you develop your package, add proper docstrings to your modules, classes, and functions
2. Run the following command to generate API documentation:

```bash
make docs-api
```
3. Change the following line in your docs/index.md file:
```md
    Replace: API Reference <api/index>
    By: API Reference <api/modules>
```
4. Run one of the following commands again:
```bash
make docs
make docs-live
make docs-clean
```
