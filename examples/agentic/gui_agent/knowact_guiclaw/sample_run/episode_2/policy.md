## Treat permission requests conservatively. Unless the current task expli...
id: guiclaw-default-conservative-permissions
type: policy
platform: all
app: 
tags: default, permissions, conservative
created_at: 1784539148.2386925
access_count: 0

Treat permission requests conservatively. Unless the current task explicitly requires and authorizes a permission, choose Deny, Cancel, Not now, or go back. If the task is blocked and authorization is unclear, request user intervention instead of granting the permission.
## When the screen is solid green or blank (1x1 placeholder), the device is...
id: guiclaw-dryrun-blank-screen-recovery
type: policy
platform: all
app: 
tags: dry-run, blank-screen, recovery, failure-avoidance
created_at: 1784539200.0
access_count: 0

When the screen is solid green or blank (1x1 placeholder), the device is in dry-run mode with no real UI. Repeatedly pressing Home will not change the screen. Instead, report the task as blocked using the done action with a clear explanation that no UI elements are visible, rather than attempting further navigation actions.
