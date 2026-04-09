# Legal Consulting Meeting Summary

## Meeting Details

- **Date:** April 8, 2026
- **Time:** 16:00-16:30
- **Participants:** Asaf, GPT-5.4
- **Purpose:** Clarify the legal and commercial constraints of building FineTree on Apache 2.0-based open-source foundation models for enterprise sales.

## Summary

This meeting focused on the legal and commercial implications of building FineTree on top of open-source foundation models, especially Apache 2.0-licensed models such as Qwen, while preserving practical commercial control in enterprise deployments. The main conclusion was that FineTree can still be sold commercially on top of an Apache 2.0 model, but fine-tuning or heavy modification does not erase the upstream open-source origin or fully convert the resulting model into something exclusively owned in the same sense as a model trained from scratch.

The discussion clarified an important distinction between open-source license rights and contract rights. Apache 2.0 continues to grant redistribution rights as a matter of license, but enterprise contracts can still restrict specific customers from reselling, redistributing, distilling, or deploying the system outside agreed conditions. Those restrictions are enforceable as contract claims against the signing customer, but they do not globally override the upstream open-source license for unrelated third parties who later obtain the model outside that contract path.

The meeting also examined branding, deployment, and technical protection. FineTree can describe the system publicly as "our model" or say that it built a model that performs a specific task, provided that it does not falsely imply original from-scratch training. On-premise API-style deployment was identified as the strongest practical compromise between enterprise privacy requirements and vendor control, while engineering protections such as encrypted weights, restricted runtimes, and API-only exposure can materially increase friction and auditability without making extraction impossible in a fully customer-controlled offline environment.

## Key Discussion Points

- Apache 2.0-based foundation models can still support a commercial FineTree product.
- A fine-tuned or heavily modified open-source model can be commercialized, branded, and sold, but the upstream lineage cannot be legally erased.
- Apache license rights and enterprise contract rights should be treated as distinct layers rather than as the same control mechanism.
- Enterprise contracts can restrict resale, redistribution, distillation, internal-use scope, and deployment location for the customer who signs them.
- Contract restrictions remain enforceable against the signing customer, but they do not nullify Apache 2.0 rights for third parties who later receive the model outside the contract.
- FineTree can market the system as "our model" or describe it as a model it built, as long as public claims do not falsely imply training from scratch.
- If model weights are distributed, attribution materials or other required notices may still allow customers to discover the upstream base model lineage.
- On-premise deployment does not necessarily require handing over raw model files or direct access to weights.
- An on-premise, API-style deployment model is a workable enterprise pattern because the model can run inside the customer's environment while exposing only inference endpoints.
- API-only deployment improves practical control compared with shipping raw weights, but it still cannot eliminate extraction risk when the customer controls the full runtime.
- Technical protections such as encryption, sharded weights, local key management, restricted runtimes, API-only interfaces, and no-plaintext persistence can raise the barrier to redistribution and casual access.
- Technical barriers can create friction and auditability, but they cannot guarantee prevention when the model runs fully offline on customer-controlled infrastructure.
- Full clean ownership would require a model trained independently from scratch rather than derived from an Apache 2.0 base model.
- Distillation from an Apache-licensed teacher remains a legally gray area rather than a clearly settled route to independence.
- A direct commercial license from an upstream model provider may be possible in principle, but there is no obligation for that provider to offer exclusive or specially restrictive rights.

## Conclusions

- FineTree can be sold commercially even if it is built on top of an Apache 2.0-licensed model.
- Enterprise contracts can restrict resale, redistribution, distillation, and deployment conditions for the customer who signs them.
- FineTree can publicly describe the system as "our model" so long as it does not misrepresent the model as trained from scratch.
- On-premise API-style deployment is a viable enterprise pattern for balancing customer privacy requirements with stronger vendor control.
- Technical protections should be viewed as practical friction and control mechanisms, not as a path to absolute prevention.

## Unresolved Issues

- No reliable path was identified to make redistribution or extraction impossible when the model runs entirely inside a customer's isolated, fully controlled environment.
- No clean legal mechanism was identified to convert an Apache-derived model into something fully and exclusively owned in the same sense as a model trained from scratch.
- Distillation remains a potentially relevant strategic path, but its legal status was not resolved in this discussion.

## Next Steps

Use these conclusions to inform future enterprise deployment strategy, commercial positioning, and customer contract design, while keeping formal legal sign-off separate from this internal discussion note.
