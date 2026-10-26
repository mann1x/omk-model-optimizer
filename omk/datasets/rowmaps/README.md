# Row maps: the upstream source of every document in a built mix

The raw upstream files behind `data/mix_v1`, `data/mix_v2` and the `mix_v2/fresh` FineWeb tail were deleted after
the mixes were built. These maps name every document they used, so any document can be fetched again exactly.

- `mix_v1.rowmap.jsonl`, `mix_v2.rowmap.jsonl` (`builders/build_mix.py --fetch-rows`)
  - One record per train and eval document: `{id: <mix>/<source>/<split>/<i>, repo, rev, path, offset, length,
    sha256, url}`.
  - `offset` and `length` are the byte range of the JSONL line in the upstream file at revision `rev`; `sha256`
    is the hash of that line.
  - Every row was fetched by HTTP range and proved: re-rendered with the mix's Qwen3.5-4B template, it reproduces
    the mix's own document byte for byte.
- `mix_v2_fresh_fineweb.rowmap.jsonl` (`builders/fresh_tail.py --rowmap`)
  - One record per FineWeb-edu document: `{id: <fresh file>/<i>, repo, rev, path, row, upstream_id, url}`.
  - `row` is the parquet row index and `upstream_id` is FineWeb's own document id.
  - The map is positional (the tail is contiguous) and every text is checked. FineWeb repeats some texts
    verbatim, so a text match alone would be ambiguous.

`<i>` is the line index in `data/<mix>/<source>.<split>.jsonl` (and in the fresh tail file).
