# Live closed-loop engineering-run ledger

This ledger prevents failed backend probes and development workloads from being
silently promoted into the prospective experiment.  Entries are immutable
after they are recorded; later runs use new IDs.

## Rejected before screening

### `fcfs_off_s24_r1`

- Date: 2026-08-16 UTC
- Role: development backend probe; FCFS, speculation off, 24 development
  sources, direct visit.
- Search path: Wikipedia MediaWiki Action API.
- Outcome: rejected.  Only 3/24 tasks succeeded; 21 task failures contain HTTP
  429 responses from Wikimedia's robot-policy limiter.
- Evidence:
  `reproduction/artifacts/live_joint/fcfs_off_s24_r1/cell/result.json`
- Evidence SHA256:
  `ea70197691ceef9bfe5dd05960a8433ec24af54ba1be65d1a80015d32118ebd1`
- Exclusion: incomplete logical tasks and calls; backend is unusable at the
  intended concurrent load.  This run is not a matrix cell, tuning result, or
  performance observation.

### `fcfs_off_rest_s24_r1`

- Date: 2026-08-16 UTC
- Role: development backend probe; FCFS, speculation off, 24 development
  sources, Wikipedia REST search/page path.
- Search path: Wikipedia REST search; direct REST page visit where reached.
- Outcome: rejected.  0/24 tasks succeeded; all 24 task failures contain HTTP
  429 responses from Wikimedia's robot-policy limiter.
- Evidence:
  `reproduction/artifacts/live_joint/fcfs_off_rest_s24_r1/cell/result.json`
- Evidence SHA256:
  `289f73771b5243f8317ee7187b90eb4e2b490067bd9620c6533856f9edb6bae3`
- Exclusion: incomplete logical tasks and calls; backend is unusable at the
  intended concurrent load.  This run is not a matrix cell, tuning result, or
  performance observation.

## Backend decision after rejection

The promotable external-live path is frozen to Bing HTML search
(`backend=bing_html_search`, `request_host=www.bing.com`) and Jina visit
(`backend=r.jina.ai`, `request_host=r.jina.ai`).  Both are real HTTP work in
the shared bounded broker.  Any fallback, backend mixture, retry, or host change
creates a new engineering run and cannot be pooled into the frozen comparison.

`live_joint_wikipedia_v1.json` is development-only because it was inspected in
these engineering runs.  Formal evaluation requires a new, untouched v2 set of
60 independent sources.
