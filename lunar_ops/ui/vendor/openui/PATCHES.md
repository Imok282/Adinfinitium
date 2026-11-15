# Local patches to the vendored OpenUI bundle

`openui-bundle.min.js` is the MIT-licensed OpenUI browser bundle (https://github.com/thesysdev/openui). Two local changes,
because the bundle does not expose OpenUI's theme provider:

1. The fallback chart palette `HEr` (a blue ramp) is replaced by the project's data colours, validated for colour-vision
   deficiency and contrast on the black chart surface (dataviz validator, dark mode, surface #000000):
   `#3987e5` blue, `#d95926` orange, `#9085e9` violet, `#c98500` yellow, `#d55181` magenta. No green: green is the UI
   chrome colour.
2. `UEr`, which spread a ramp from its middle outward, now hands out the palette in fixed order (series 1 = blue,
   series 2 = orange, ...), so a series keeps its colour when others are added or filtered out.

## Export the full component library (2026-11-15)

`window.__OpenUI` now also exports `openuiLibrary` (the bundle's full library, minified name `NJa`) next to
`openuiChatLibrary`. The chat library has no `Stack` layout component; `ouichart.js` switches to the full library
only when a chart program uses `Stack(...)` (side-by-side small multiples).
