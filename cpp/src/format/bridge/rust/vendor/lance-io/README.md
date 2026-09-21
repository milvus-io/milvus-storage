# Lance scheduler request context hook

The two scheduler source files (including their upstream tests) come from
`zilliztech/lance` at `d512a02e636057fc9e3bdd2304548e19bbd1a369`.
All other modules and types are re-exported from that exact upstream crate.

The only scheduler change is `new_with_reader_wrapper`: a stateless function
wraps the Reader at each `submit_request`, before either scheduler queues work.
Storage uses this to capture request context; the wrapper is retained only by
that request. This preserves sharing, priorities, cancellation, and AIMD policy.
The standard scheduler remains the default. No thread-local parent is retained
by the shared scheduler or ObjectStore.

Remove this facade once the pinned upstream exposes an equivalent request hook.
Keep upstream scheduler tests when refreshing these files.
