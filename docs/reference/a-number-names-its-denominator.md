# A number that names a quantity must name its denominator

Three times in one week, two denominators carrying the same name cost us a
decision. This is no longer a distraction, it is a class.

| the quantity | the two denominators | what it cost |
|---|---|---|
| "the hub's models" | **56** artefacts measurable from this workstation / **47** catalogue models | two tables placed side by side that were not counting the same population |
| "the certified entries" | **971** / **1,509** | a comparison between two machines that was not about the same thing |
| "the free memory" | **0.1 GB** (`Pages free`) / **9.4 GB** (`available_mb`) | a wait decided on the wrong grounds |

## The rule

**A number that names a quantity names its denominator, in the same
sentence.** "22 are running" means nothing; "22 of the 56 artefacts
measurable from this workstation" means something.

## And the corollary that decides when two sources exist

**When one of our own modules computes this quantity, IT is the authority,
not the system tool.**

One of our own modules carries the question in its definition. `vm_stat`
answers its own question, which is not ours: `Pages free` on macOS is almost
always tiny because the system does not keep free pages — the real space is
in the inactive and purgeable pools. Reading `Pages free` and calling it "the
free memory" borrows a tool's word for a question it does not ask.

`core/host_memory.py` computes `available_mb`, and that is the number Prism
plans against. It is therefore the one that decides, and the one we cite.

**Second trap in the same place:** this machine's page size is **16,384
bytes**, not 4,096. A page-based calculation that assumes 4 KB is off by a
factor of four, on top of answering the wrong question. A module that reads
the page size from `sysctl` cannot make this mistake; a hand calculation
makes it every time.

## The practice

1. **Write the denominator with the numerator.** Always, including in a
   transitional sentence.
2. **Prefer our module over the system tool** when both exist, and say which
   one was read.
3. **When a number motivates a decision, say what it motivates.** "I'm
   waiting for lack of memory" and "I'm waiting because the measurement is
   serialized" are resolved differently: the first by cutting something, the
   second by waiting one's turn. We do not cut someone's work for the wrong
   reason.

This third point is what almost happened here: the virtual machine is
running, 5.3 GB and three quarters of a core, and **9.4 GB remain
available**. What was causing the wait was **GPU serialization**, not
memory.
