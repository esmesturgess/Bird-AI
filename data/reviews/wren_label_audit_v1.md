# Wren Label Audit v1

This table audits the Wren semantic buckets we selected from Xeno-canto.
It keeps the raw `sound_type` and `stage` values and marks whether our chosen bucket looks clean, mixed, or questionable.

- Full CSV: `/Users/esmesturgess/Bird Translation AI/data/reviews/wren_label_audit_v1.csv`
- Total rows: `44`

## Summary by bucket and mapping quality

- `alarm`: clean=4, mixed=4, questionable=2
- `call`: clean=12
- `juvenile_begging`: clean=6, mixed=5
- `song`: clean=11

## Questionable or mixed rows

| source_set | xc_id | assigned_bucket | sound_type_raw | stage_raw | mapping_quality | note |
|---|---:|---|---|---|---|---|
| reference_pull_v2 | 1044015 | alarm | alarm call, call | uncertain | mixed | alarm bucket but raw sound_type mixes alarm with call / flight / rattle content |
| reference_pull_v2 | 448884 | alarm | alarm call, call, flight call, outburst in flight and rattle | nan | mixed | alarm bucket but raw sound_type mixes alarm with call / flight / rattle content |
| reference_pull_v2 | 481873 | alarm | alarm call, call | juvenile | mixed | alarm bucket but raw sound_type mixes alarm with call / flight / rattle content |
| reference_pull_v2 | 799671 | alarm | alarm call, call, rattle | adult | mixed | alarm bucket but raw sound_type mixes alarm with call / flight / rattle content |
| reference_pull_v2 | 717599 | alarm | call, agitation/agression calls | nan | questionable | alarm bucket built from a neighboring behaviour label rather than explicit alarm |
| reference_pull_v2 | 798453 | alarm | call, rattle | adult | questionable | alarm bucket built from a neighboring behaviour label rather than explicit alarm |
| holdout_balanced_v1 | 807457 | juvenile_begging | call | juvenile | mixed | juvenile stage is clear but sound_type is only generic call |
| reference_pull_v2 | 181392 | juvenile_begging | call | juvenile | mixed | juvenile stage is clear but sound_type is only generic call |
| reference_pull_v2 | 188239 | juvenile_begging | call | juvenile | mixed | juvenile stage is clear but sound_type is only generic call |
| reference_pull_v2 | 379435 | juvenile_begging | call | juvenile | mixed | juvenile stage is clear but sound_type is only generic call |
| reference_pull_v2 | 807457 | juvenile_begging | call | juvenile | mixed | juvenile stage is clear but sound_type is only generic call |