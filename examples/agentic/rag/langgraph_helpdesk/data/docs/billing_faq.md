# Billing and Usage FAQ

Answers to common billing questions for CVlization platform adopters.

## Do I need a license to run examples?

No additional license is required; each example honors the upstream project's license.

## How do I estimate cloud costs?

- Build Docker images locally before running in the cloud.
- Check GPU time requirements in each example's README.
- Use spot instances for long training runs when possible.

## Can I share models across teams?

Yes. Configure a shared network volume for `~/.cache/cvlization` so teams reuse downloaded weights.

