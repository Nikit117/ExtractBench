# Verifying a submitted row

The row schema is documented in [ExtractBench: submitting results](extractbench.md#submitting-results),
and the shared steps are in the [README](../README.md#submitting-results). This
page covers the part those two do not: **how a submitted row gets checked, what
makes that check cheap, and what usually causes a round trip.**

It is written from the submitter's side, because that is where the information
is. Most of it is advice a submitter can act on before opening the pull
request.

## What checking a row actually costs

Checking a row has two parts, and they are not equally expensive:

| part | what happens | cost |
|---|---|---|
| scoring | the official scorer runs over the submitted predictions | minutes |
| running | the pipeline is run over the corpus and the predictions are produced | the expensive part |

A full run is single-digit hours by the [inclusion criteria](../README.md#submitting-results),
so **the run is what a reviewer spends time on**, and it is also the part that
needs an API key, a model endpoint and possibly a second service. Scoring, by
contrast, is cheap and deterministic.

Everything below is about shifting work from the second row to the first.

## Publish the predictions, not only the number

The single most useful thing a submitter can attach is the **per-document
predictions** of the submitted run, in the format the scorer consumes, together
with the per-document official scores.

With those two things, a reviewer can:

- recompute the score without running anything and without spending an API key
- check any subset of documents instead of the whole corpus
- compare a fresh run against the published numbers document by document
- confirm that the score belongs to the published predictions, rather than to a
  run nobody can see

A useful shape, and the one we used in
[#84](https://github.com/run-llama/ExtractBench/pull/84):

```
predictions.zip          <doc_id>/prediction.json for every document
scores.csv               doc_id,value_f1[, precision, recall] per document
SHA256SUMS.txt           sha256 of each file above
```

The hash manifest is cheap and removes the "did the file change" question
entirely. The per-document distribution matters more than the mean: one run of
an agentic pipeline is one draw, not a deterministic function, so the mean alone
cannot tell a reviewer whether a fresh run is consistent or broken.

## Say what the run needs, before the reviewer has to ask

A reviewer cannot guess any of these, and each one that is unstated is a round
trip. State them in the pull request:

- **the endpoint, the model and its version** -- for example `gpt-5.4` on Azure
  OpenAI, or `gpt-5.4` on a standard Chat Completions endpoint
- **every service the run needs, not only the model.** A run that also uses a
  layout or OCR service needs two credentials, and the score can depend on the
  second one. Say so, and say what the score looks like without it
- **the exact configuration switches** the run used, with their values
- **the cost derivation** -- tokens or pages times the public list price, the
  number of pages, and explicitly whether any free tier or credit was assumed.
  The README asks for public prices intended to remain in effect indefinitely,
  excluding discounts and free credits, so the derivation should say the same
- **what is published and where**, with links, not descriptions

## Pin the revision in a way that cannot move

A commit hash on its own is not a pin: a branch can be rewritten and the hash
can stop resolving to the code that produced the numbers. A **tag** survives
that, because tags are not moved by a force-push.

So publish the commit the numbers were produced from, and give it a tag that
will not be moved. If the submission points at a third-party repository, pin
that repository the same way, and say in the pull request which revision the
run used.

## Make a subset runnable in one command

A full run is single-digit hours, which is long enough that most reviewers will
not start one. Offer a subset instead:

```bash
# the same pipeline, the same switches, N documents
python reproduce.py --docs 5
```

Printing the published per-document score next to each document that just ran
turns the command into a comparison, not just a run. A reviewer who can check
five documents in a few minutes is far more likely to finish the review than
one who is asked to commit to the whole corpus.

## What caused a round trip, measured

From our own submission, which took more than one:

- a score with no per-document numbers behind it -- the reviewer cannot tell a
  regression from a normal draw
- two required services, one of which was not mentioned in the first description
- a cost per page that did not match the derivation, because the derivation was
  never written down
- a claim about a second service that was later corrected by measurement. The
  correction cost more credibility than the original claim would have

## Checklist for a submitter

- [ ] the row carries every column of the [schema](extractbench.md#submitting-results)
- [ ] the per-document scores are published as a CSV
- [ ] the predictions are published as a zip, in the format the scorer consumes
- [ ] a hash manifest covers the published files
- [ ] the revision is pinned by commit **and** by a tag that will not move
- [ ] a one-command subset run is available, and prints the published scores
- [ ] the cost derivation is written out, with the price source
- [ ] the endpoint, the model version and every other service are named
- [ ] the parts that reproduce without a paid service are named, with their
      measured numbers

## The shortest path for a reviewer

1. download the published predictions and check the hash manifest
2. run the one-command subset and compare it with the published per-document
   scores
3. confirm that the pinned tag still resolves to the revision that was published
4. check the cost derivation against the row
5. run the full corpus only if the subset and the numbers disagree
