# PyPI Credential Management

This document explains how this project manages **PyPI** and **Test PyPI** credentials and make them available for **Poetry**. The focus is on **safe, explicit, and predictable credential handling**, without forcing tool-specific secret names on users.

---

## Goals

* Allow per-project credentials
* Avoid committing secrets
* Avoid accidental publishing with the wrong account
* Work locally and in CI
* Keep credentials **provider-centric**, not tool-centric
* **Poetry specific credentials** are resolved at runtime and passed only to the command that needs them

---

## Credential sources and resolution order when publishing manually (eg. poetry publish or make publish)

Credentials are resolved in the following order:
1. Project-local `.env` file
2. Environment variables
3. Poetry global configuration (`auth.toml`)

This order ensures that:
* Projects can define their own credentials
* Global configurations are used only as a fallback

---

## Credential source when publishing via CI (push tags v*)

* Credentials for CI are always sourced from repo secrets
* The repo secrets can be created automatically by the use of the --secrets flag
* The --secrets option sources credentials in the following order:
  1. `.env` file found in the project path
  2. Environment variables
* Poetry specific environment variables and global configuration are not sources of credentials for CI workflow execution
* CI workflow will inject secrets explicitly as environment variables at runtime

---

## Tool-agnostic secret names

Secrets are named after the **service**, not the publishing tool:
* PYPI_TOKEN
* TEST_PYPI_TOKEN

These names:

* Are easy to understand
* Work across tools
* Avoid leaking tool-specific conventions into project configuration

Mapping to Poetry-specific variables happens **only at execution time**.

---

## The `.env` file

Each project may optionally define a `.env` file at the repository root.

Example contents (values shown here are placeholders):

```
PYPI_TOKEN=pypi-xxxxxxxxxxxxxxxx
TEST_PYPI_TOKEN=pypi-yyyyyyyyyyyyyyyy
```

Important properties:

* The file is not committed
* It is treated as data, not executable code
* Values may be quoted or unquoted
* Only explicitly requested keys are read

The `.env` file must be added to `.gitignore`.

---
