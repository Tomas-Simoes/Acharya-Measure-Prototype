# Universal Pepe / Claudino Protocol — Single Source of Truth (v7.0)
# Last updated: 2026-05-24
# Applies to: ALL agents — Kimi (Maestro), Claude Code, OpenCode (Claudino)
#
# Kimi is the LEAD AGENT (Maestro). All other agents report to Kimi.
# If local rules in .ai/protocol.md conflict with universal rules, LOCAL wins.

---

## 🎭 Agent Hierarchy

| Agent | Role | Tool |
|-------|------|------|
| **Kimi** | Maestro / Orchestrator | `pep` / `pepe` |
| **Claude Code** | Specialist | `claude` (Anthropic CLI) |
| **Claudino (OpenCode)** | Specialist | `cld` / `claudino` |

**All agents report to Kimi.** After every session, write to `.ai/history.md`, update `.ai/tasks.md`, and add a handoff note under `## 📬 Reports to Maestro` in `.ai/tasks.md`.

---

## 🔴 RED LINES — ABSOLUTE PROHIBITIONS

1. **NEVER push to `main` or `master` without Pedro explicitly saying "push to main"**
2. **NEVER push broken builds** — run the full quality gate before EVERY commit
3. **NEVER commit without updating `.ai/` files** — `history.md`, `tasks.md`, `plan.txt` minimum
4. **NEVER implement business logic without asking Pedro first**
5. **NEVER delete files** — ask first
6. **NEVER modify `package.json` dependencies** — ask first
7. **NEVER change `.env` variables** — ask first
8. **NEVER assume when you have a doubt — ASK PEDRO FIRST.** Stop. Ask. Do NOT guess.
9. **NEVER start coding without doing the Startup Protocol first**

---

## ✅ Startup Protocol (EVERY session, in order)

Run these commands first:
```sh
git branch --show-current
git log --oneline -5
```

Then read in order:
1. `.ai/protocol.md` — this file (you are reading it now ✅)
2. `.ai/context.md` — project overview, stack, key paths
3. `.ai/plan.txt` — current roadmap and priorities
4. `.ai/tasks.md` — what is pending and in progress
5. `.ai/history.md` (last 30 lines) — what happened last session
6. `.ai/decisions.md` — architectural decisions already made
7. `.ai/learnings.md` — patterns discovered, gotchas, validated rules
8. `.ai/metrics.md` — what approaches work best
9. `.ai/kimi-maestro.md` — Kimi strategy file (delegation rules)

Then **confirm with Pedro** before touching anything:
> "I'm on branch `<branch>`. Last commit: `<message>`. Ready for instructions."

**Do NOT start coding before Pedro responds to this confirmation.**

---

## 🧠 Memory Write Protocol (prevents race conditions)

When writing to any `.ai/` file:
1. Check if `.ai/.memory-lock` exists — if yes, wait 1s, retry up to 3 times
2. Write your PID to `.ai/.memory-lock`
3. Read current file, append/update, write
4. Remove `.ai/.memory-lock`, update `.ai/.last-writer` with your agent name

---

## ✅ Before You Write Any Code — Mandatory Pre-Coding Check

1. **State what you understood** — repeat back what Pedro asked for
2. **State what files you will change** — list them explicitly
3. **State any assumptions** — if you have ANY, that is a question for Pedro
4. **Wait for Pedro to say "yes, go ahead"**

---

## 🧮 Quality Gate (run before EVERY commit, in order)

1. `npm run lint` — ESLint + Prettier (zero errors)
2. `npx tsc --noEmit` — TypeScript (zero type errors)
3. `npm test` — all tests pass
4. `npm run build` — build succeeds
5. Responsive check — verify UI at 1280px, 768px, 375px

If any step fails: STOP, fix, re-run all from step 1. Never commit with failures.

---

## ✅ Before Every Commit — Mandatory Pre-Commit Checklist

```
[ ] Quality gate passes (all 5 steps)
[ ] Committing to correct branch (NOT main/master unless Pedro said so)
[ ] .ai/history.md updated with what was done and why
[ ] .ai/tasks.md updated (completed items moved, new items added)
[ ] .ai/plan.txt updated (completed steps marked [x])
[ ] .ai/metrics.md updated with session data
[ ] .ai/learnings.md updated if new patterns discovered
[ ] No files with secrets added (.env, credentials)
[ ] git add lists only files I intentionally changed
```

---

## 🚀 Commit & Push Flow

```sh
git add <specific files>          # NEVER "git add -A" blindly
git commit -m "type(scope): what happened and why"
git push origin <branch>
```

Types: `feat | fix | refactor | style | test | docs | chore`

**Never end a session without pushing.** A commit that is not pushed does not exist for Pedro.

---

## 🚀 Deployment Rules — Where Code Goes Live

| Environment | What it is | Can agent use it? |
|-------------|-----------|-------------------|
| **Localhost** | `npm run dev`, `next dev`, `vercel dev` — runs on YOUR machine only | ✅ Yes — safe for development and testing |
| **Preview / Staging** | Auto-deployed by Vercel/Netlify when you push to `dev` | ✅ Yes — safe for testing and sharing |
| **Production** | The live public URL (`*.com`, `*.vercel.app` without preview prefix) | 🚫 NO — Pedro only |

### Local Development (✅ Allowed)
- `npm run dev` / `next dev` / `vercel dev`
- `npm start` (if it starts a local server)
- Any command that runs on `localhost:*` or `127.0.0.1:*`
- These are PRIVATE to your machine. Safe to run anytime.

### Preview / Staging (✅ Allowed)
- Pushing to `dev` triggers automatic preview deployment
- Vercel gives you a URL like `https://project-git-dev-username.vercel.app`
- Netlify gives you a deploy preview URL
- Test here. Share this URL with Pedro for review.

### Production (🚫 FORBIDDEN)
- The live public URL that users see
- Vercel: `vercel --prod`
- Netlify: `netlify deploy --prod`
- Any command with `--prod`, `--production`, or that deploys to the live URL
- Changing the production branch in dashboard from `main` to `dev`

### Agents MUST do this:
- Push to `dev` after every commit
- Let the platform auto-deploy `dev` to a preview URL
- Test on **localhost** during development
- Test on **preview URL** before asking Pedro to review
- NEVER test on the production/live URL

### If Pedro explicitly says "deploy to production":
1. Confirm back: "You want me to deploy branch `<branch>` to production?"
2. Wait for Pedro to confirm with "yes"
3. Run the production deploy command
4. Verify the live site works
5. Report back to Pedro with the live URL

**If you are unsure whether a command deploys to production: STOP and ask Pedro.**

---

## 📚 Learning Protocol

When you discover a pattern or solve a novel problem:
1. Add entry to `.ai/learnings.md` with date, context, approach, success rating
2. Before starting similar tasks, check `learnings.md` for relevant entries

---

## ✅ Exit Protocol — Triggered by "cld", "claudino", "exit", or "stop"

When Pedro says to end the session:

1. Acquire lock (`.ai/.memory-lock`)
2. Append to `.ai/history.md` — dated entry, bullet points of what was done and why
3. Update `.ai/plan.txt` — mark completed steps `[x]`, update status
4. Update `.ai/tasks.md` — move completed items, add newly discovered tasks
5. Update `.ai/metrics.md` with session data
6. Update `.ai/decisions.md` — record any architectural decisions made
7. Update `.ai/learnings.md` — record any new patterns or gotchas discovered
8. Release lock

Then reply: **"✓ Session saved."** and stop.

---

## Protected Actions (require Pedro's explicit instruction)

- Merging into `main` or `master`
- Deleting any file
- Modifying `package.json` dependencies
- Changing `.env` or environment variables

---

## `.ai/` Files — Update Every Session

| File | When to update |
|------|----------------|
| `history.md` | After every session — append dated entry |
| `tasks.md` | Mark completed tasks, add newly discovered tasks |
| `plan.txt` | Mark completed steps `[x]`, update status |
| `decisions.md` | Record any architectural or design decision made |
| `learnings.md` | Add patterns, gotchas, validated rules |
| `metrics.md` | Session data, what approaches work best |

These are Pedro's memory across all AI tools. Stale files = broken context = wasted sessions.
