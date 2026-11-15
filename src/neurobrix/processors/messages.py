"""Every sentence a person reads about a model's processors, as templates, and nowhere else.

A processor is a declared, non-neural sub-component of a model (a text front end, for one). What the person sees of
it is two lines at its first run, a bar on a terminal, and, when something is wrong, ONE sentence that says what to
do. No URL, no log line and no stack reaches them: the cause is chained to the refusal, shown under `-vv` only.

`{label}`, `{size}`, `{file}`, `{name}`, `{version}`, `{variants}` come from the model's declaration; `{model}` is
the name the person typed.
"""
from __future__ import annotations

#: the command that holds the processors (open question F10 of the design: the one place its name is written)
COMMAND = "processor"

BEFORE = "This model needs the {label} sub-component. Downloading it once ({size})..."
READY = "{label} ready, checksum verified"
UNREACHABLE = ('This model needs the {label} sub-component and this machine could not reach the download. '
               'On a connected machine run "neurobrix ' + COMMAND + ' fetch {model} --to <folder>", copy {file} here, '
               'then run "neurobrix ' + COMMAND + ' install {file}".')
MISMATCH = ("The {label} sub-component that arrived is not the one this model declares (its checksum differs from "
            "the one the model's container configuration states). Nothing was installed. Run the command again; "
            "if it happens twice, write to contact@neurobrix.org.")
UNPUBLISHED = ('This model needs the {label} sub-component ({name} {version}), which is not published where this '
               'model says. Update NeuroBrix, or install it from a file with "neurobrix ' + COMMAND
               + ' install <file>".')
TOO_NEW = "This model needs a newer NeuroBrix to run its {label} sub-component. Update NeuroBrix."
CUT = ("The download of the {label} sub-component stopped before the end. Run the command again: it continues "
       "where it stopped.")
VARIANT = 'This model\'s {label} sub-component does not read "{variant}". It reads: {variants}.'
NO_COMPONENT = ("This model's {label} sub-component needs the model's {component} part, which this container does "
                "not hold. Import the model again.")
NOT_LENT = ("This model's {label} sub-component asks the model's {component} part, and this version of NeuroBrix "
            "does not run it for a sub-component yet. Update NeuroBrix.")
NOT_DECLARED = ('{file} is not a sub-component the model {model} declares. It declares: {files}.')
NONE_DECLARED = "The model {model} declares no sub-component: it needs nothing more."
NOT_A_FILE = 'There is no file at {path}. Give the path of the file "neurobrix ' + COMMAND + ' fetch" wrote.'


class ProcessorRefused(ValueError):
    """A processor this model declares cannot be given to it: `str()` is the one sentence the person reads (a
    ValueError, as the command line's other named refusals: said in one line, the cause under `-vv` only)."""


def refusal(template: str, **fields) -> ProcessorRefused:
    """The refusal whose sentence is `template` filled with `fields` (raise it `from` its cause)."""
    return ProcessorRefused(template.format(**fields))
