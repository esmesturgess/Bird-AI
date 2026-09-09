# Bird Translation AI — sound artist pack

Twelve clips, one per species x vocalisation type, all classified correctly by the model.
Organised as `<species>/<species>_<vocalisation>.wav`.

    blackbird/   song, call, alarm, juvenile
    robin/       song, call, alarm, juvenile
    tawny_owl/   song, call, alarm, juvenile

`manifest.csv` has full attribution per clip: recordist, source archive, a link to the
original listing, and the source's own license code.

## Two things to know before using these publicly

**Licenses are recorded as-is, not decoded.** Xeno-canto clips link straight to a
Creative Commons license (readable at the URL given). Macaulay Library clips carry an
internal code like `LICENSE4` that we could not resolve programmatically — the asset
pages are JavaScript-rendered and did not return readable text when checked. Open the
`source_url` for each Macaulay clip in a normal browser and read the license shown there
before using it beyond this internal review.

**Two clips were seen by the model during training**, flagged in
`model_was_trained_on_this_recording`: blackbird juvenile and robin juvenile. Every
other combination of species and juvenile call was tried and the model gets it wrong, so
these are the only examples that could be included at all — worth knowing if the honesty
of "the machine correctly identifies this" matters to how you treat them.

## Audio note

Four of the twelve source clips are shorter than the installation's 25-second slot
(`looped_to_fill_25s = True` in the manifest) and are looped, with a 0.4s gap between
repeats, to fill the full slot rather than trailing into silence. If you are working from
these files directly rather than the assembled soundscape, you may want the original
un-looped recording instead — ask and I can provide it.
