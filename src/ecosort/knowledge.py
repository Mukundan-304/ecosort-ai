"""Item-level waste knowledge base: material kind, recyclable?, biodegradable?, hazard?, handling, recovery stream."""
def _i(kind, stream, rec, bio, kg, tip, icon, prompt, hazard=False):
    return dict(kind=kind, stream=stream, recyclable=rec, biodegradable=bio, hazard=hazard, kg=kg, tip=tip, icon=icon, prompt=prompt)
ITEMS = {
 "plastic bottle":   _i("Plastic (PET)", "plastic_pet",  True,  False, .03, "Empty, rinse, remove cap, crush. Blue bin.", "🔵", "a plastic bottle"),
 "plastic container":_i("Plastic (HDPE/PP)", "plastic_hdpe", True, False, .06, "Rinse, keep lid on. Blue bin.", "🔵", "a plastic container or cup"),
 "plastic bag":      _i("Plastic film", "residual",      False, False, .01, "Not curbside-recyclable. Bundle separately -> waste-to-energy.", "🟣", "a plastic bag or wrapper"),
 "styrofoam":        _i("Polystyrene foam", "residual",  False, False, .02, "Not recyclable. Keep dry, send to waste-to-energy.", "🟣", "styrofoam packaging"),
 "glass bottle":     _i("Glass", "glass",                True,  False, .35, "Do not break. Rinse. Green bin.", "🟢", "a glass bottle or jar"),
 "aluminum can":     _i("Metal (aluminium)", "metal",    True,  False, .015, "Empty and crush lightly. Grey bin.", "⚪", "an aluminum drink can"),
 "metal tin can":    _i("Metal (steel)", "metal",        True,  False, .05, "Empty, rinse, press lid inside. Grey bin.", "⚪", "a metal tin can"),
 "paper":            _i("Paper", "paper",                True,  True,  .02, "Keep DRY, no food residue. Yellow bin.", "🟡", "a sheet of paper"),
 "newspaper":        _i("Paper", "paper",                True,  True,  .15, "Keep dry and bundled. Yellow bin.", "🟡", "a newspaper"),
 "cardboard box":    _i("Cardboard", "paper",            True,  True,  .10, "Flatten, keep dry. Yellow bin.", "🟡", "a cardboard box"),
 "banana peel":      _i("Organic", "organic",            False, True,  .10, "Compost / biogas stream. Keep out of dry bins.", "🟤", "a banana peel"),
 "food waste":       _i("Organic", "organic",            False, True,  .25, "Compost / biogas stream. Drain liquids.", "🟤", "leftover food"),
 "fruit peel":       _i("Organic", "organic",            False, True,  .08, "Compost / biogas stream.", "🟤", "an orange or fruit peel"),
 "leaves":           _i("Organic", "organic",            False, True,  .05, "Compost stream.", "🟤", "dry leaves or plant waste"),
 "eggshell":         _i("Organic", "organic",            False, True,  .01, "Compost stream.", "🟤", "an eggshell"),
 "battery":          _i("Hazardous (battery)", "ewaste", False, False, .05, "HAZARD: gloves, never crush or bin. Red e-waste cage.", "🔴", "a battery", True),
 "electronic device":_i("E-waste", "ewaste",             False, False, .30, "HAZARD: gloves, do not crush. Red e-waste cage.", "🔴", "an electronic device or mobile phone", True),
 "cloth":            _i("Textile", "textile",            True,  False, .20, "Keep dry and clean. Textile bag.", "🩷", "a piece of clothing or cloth"),
}
CLASSES = list(ITEMS)
