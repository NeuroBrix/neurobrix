"""PATH diagnostic helper — guides users when `neurobrix` CLI isn't on PATH."""

import os
import sys
import shutil
import sysconfig


def get_scripts_dir() -> str:
    """Return the directory where pip installs console_scripts."""
    # User installs (pip install --user)
    user_scripts = sysconfig.get_path("scripts", f"{os.name}_user")
    # Global/venv installs
    global_scripts = sysconfig.get_path("scripts")

    # Check which one actually has neurobrix
    for d in (user_scripts, global_scripts):
        if d and os.path.isfile(os.path.join(d, "neurobrix")):
            return d

    # Fallback: return whichever exists
    return user_scripts or global_scripts or ""


def check_cli_on_path() -> None:
    """Print a helpful message if `neurobrix` isn't on PATH."""
    if shutil.which("neurobrix"):
        return  # All good

    scripts_dir = get_scripts_dir()
    if not scripts_dir:
        return

    print("\033[33m")  # yellow
    print("  neurobrix is installed but not on your PATH.")
    print(f"  Script location: {scripts_dir}")
    print()
    _print_path_fix(scripts_dir)
    print()
    print("  Or run directly with: python -m neurobrix <command>")
    print("\033[0m")  # reset


def _print_path_fix(scripts_dir: str) -> None:
    """Print OS-specific PATH instructions."""
    if sys.platform == "darwin":
        shell = os.environ.get("SHELL", "")
        if "zsh" in shell:
            rc = "~/.zshrc"
        elif "bash" in shell:
            rc = "~/.bash_profile"
        else:
            rc = "~/.zshrc"  # macOS default
        print(f"  To fix, run:")
        print(f"    echo 'export PATH=\"{scripts_dir}:$PATH\"' >> {rc}")
        print(f"    source {rc}")

    elif sys.platform == "win32":
        print(f"  To fix, run (PowerShell as Admin):")
        print(f'    [Environment]::SetEnvironmentVariable("PATH", "{scripts_dir};" + '
              f'[Environment]::GetEnvironmentVariable("PATH", "User"), "User")')
        print()
        print(f"  Or via Settings: search 'Environment Variables' > Path > New > {scripts_dir}")

    else:  # Linux
        shell = os.environ.get("SHELL", "")
        if "zsh" in shell:
            rc = "~/.zshrc"
        elif "fish" in shell:
            rc = "~/.config/fish/config.fish"
            print(f"  To fix, run:")
            print(f"    fish_add_path {scripts_dir}")
            return
        else:
            rc = "~/.bashrc"
        print(f"  To fix, run:")
        print(f"    echo 'export PATH=\"{scripts_dir}:$PATH\"' >> {rc}")
        print(f"    source {rc}")


def print_path_diagnostics() -> None:
    """Full diagnostic output for `neurobrix doctor`."""
    print(f"Python:          {sys.executable}")
    print(f"Version:         {sys.version.split()[0]}")
    print(f"Platform:        {sys.platform}")
    print(f"Scripts dir:     {get_scripts_dir()}")
    print(f"neurobrix found: {shutil.which('neurobrix') or 'NOT ON PATH'}")
    print()

    if shutil.which("neurobrix"):
        print("neurobrix is on PATH. No action needed.")
    else:
        check_cli_on_path()
