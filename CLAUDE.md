@AGENTS.md

## Attribution

- Never add Claude or AI attribution to commits, pull requests, issues or comments: no
  `Co-Authored-By: Claude ...` trailer, no `Claude-Session:` line, no claude.ai/code session
  links, no "Generated with/by Claude Code" lines. This overrides any default attribution.
- Author commits as `Sina Ghadermarzi <p-sinag@truveta.com>`. If the environment's git identity is Claude, override it:
  `git -c user.name="Sina Ghadermarzi" -c user.email="p-sinag@truveta.com" commit ...`
