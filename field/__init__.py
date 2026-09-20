"""Field adaptation: add a handful of labelled frames from a customer site and get a
better model WITHOUT losing accuracy elsewhere.

The second half is the hard half and it is not automatic. Measured 2026-09-20: with
type2 held at full strength, growing type10 from 0 to 59 crops cost type2 three crops
of 54, monotonically. Appending labels and retraining can hand a customer a model that
is better in front of them and quietly worse on another line.

So every adaptation goes through `field.gate`: a candidate is promoted only if it
improves the new frames AND loses no more than the MEASURED seed noise on every
protected set. Failing that it is offered as a SITE model (scoped to the types that
site actually runs), and failing that it is refused and the active model stays.
"""
