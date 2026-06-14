# rule.md — stub-naming convention (rust)

Any deliberately-incomplete code MUST be labelled so it is trivially
greppable. The frozen checker flags these; an unlabelled wrong impl
is left to the frozen tests + mutation.

- **Sentinel comment:** mark the placeholder line with `RATCHET:STUB`.
- **Sentinel symbol prefix:** name any placeholder item `rgstub_…`.
- The honest-stub macros `todo!()` and `unimplemented!()` are also
flagged; prefer not to leave placeholders at all.

Better than labelling a stub: do the real thing, or log a blocker
task and stop (CLAUDE.md rule 1).
