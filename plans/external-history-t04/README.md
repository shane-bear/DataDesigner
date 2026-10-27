# External-history call boundary

Continue the reviewed offline seed milestone with a consumer integration, using
ModelRegistry's existing facade factory and native request admission. Do not
change engine scheduling, content truth, or training admission. Test actual native
HTTP adapters and ModelFacade with synthetic inputs only.

The boundary reserves a frozen worst-case token-price allowance before each
call, fsyncs request identity, disables both executor and transport retries, and
retains first responses or uncertainty. Unknown usage, cancellation, interrupted
calls and unreturned responses retain their reservation and stop new calls.
An append-only receipt artifact can be reconciled from a single batch of
provider-side evidence; it never replays POSTs or replaces a first answer.

Prices and token ceilings are supplied assumptions, not provider guarantees.
If reported usage exceeds a ceiling, retain the response, record the excess and
stop. Receipts contain authorized request/response bodies, not HTTP headers.
Linux file locking is explicit for this research integration. No new scheduler,
model API calls, credentials, raw-history access, merge or deployment is involved.
