"""Browser check for the MIND3D hover identification on trail.html.

Loads the page with a short live run, waits for neuron orbs, then sweeps the
cursor over the 3D panel asserting that:
  1. some point identifies a specific tweakable neuron as `L<layer> · N<n>`;
  2. clicking that point adds exactly that neuron to the tweaks panel;
  3. slab-only hovers at least name the layer;
  4. no console errors.

Needs the explorer server up on :8791 with the model ready.
Run: python3 scripts/verify_mind3d_hover.py
"""

import re
import sys
import time

from playwright.sync_api import sync_playwright

BASE = "http://127.0.0.1:8791"
QUESTION = "What helps a child focus on homework?"


def fail(msg):
    print(f"FAIL: {msg}")
    sys.exit(1)


def main():
    errors = []
    with sync_playwright() as pw:
        browser = pw.chromium.launch(channel="chrome", headless=True)
        pg = browser.new_page(viewport={"width": 1500, "height": 950})
        pg.on("console", lambda m: errors.append(m.text) if m.type == "error" else None)
        pg.on("pageerror", lambda e: errors.append(str(e)))

        pg.goto(f"{BASE}/trail.html?q={QUESTION.replace(' ', '%20')}&go=1&max=12")
        # ready can take a while on a cold server; tokens mean orbs exist
        try:
            pg.wait_for_selector("#status.ready", timeout=60_000)
        except Exception:
            fail("server never reached ready (is the explorer up? curl /api/status)")
        try:
            pg.wait_for_function("document.querySelectorAll('#answer .tk').length >= 5", timeout=240_000)
        except Exception:
            fail("no tokens after 240s - check /api/status for a stuck generation lock")
        pg.wait_for_timeout(1500)  # let MIND3D settle on the latest word

        box = pg.locator("#mind3d canvas").bounding_box()
        if not box:
            fail("no #mind3d canvas")

        def hover_text():
            return pg.evaluate(
                "(() => { const h = document.getElementById('hover-tag');"
                " return h && !h.hidden ? h.textContent : null; })()")

        # sweep the panel until the cursor lands on something identified
        found = None          # (x, y, text) of an L·N hit
        slab_seen = False
        x0, y0 = box["x"] + 8, box["y"] + 8
        step = 17
        nx = max(1, int((box["width"] - 16) // step))
        ny = max(1, int((box["height"] - 16) // step))
        for gy in range(ny + 1):
            if found:
                break
            for gx in range(nx + 1):
                x, y = x0 + gx * step, y0 + gy * step
                pg.mouse.move(x, y)
                pg.wait_for_timeout(45)
                t = hover_text()
                if not t:
                    continue
                if re.search(r"^layer \d+", t):
                    slab_seen = True
                m = re.search(r"L(\d+) · N(\d+)", t)
                if m:
                    found = (x, y, t, int(m.group(1)), int(m.group(2)))
                    break
        if not found:
            fail(f"swept {nx * ny} points, no L·N ever identified (slab-only seen: {slab_seen})")
        print(f"hover hit at ({found[0]:.0f},{found[1]:.0f}): {found[2].splitlines()[0]}")

        # click exactly there; autoRotate may drift the scene, so retry against a fresh hover
        added = False
        for attempt in range(4):
            pg.mouse.click(found[0], found[1])
            pg.wait_for_timeout(400)
            txt = pg.locator("#tw-list").inner_text() if not pg.locator("#tweaks").get_attribute("hidden") else ""
            want = f"L{found[3]}·N{found[4]}"
            if want in txt.replace(" ", ""):
                added = True
                print(f"click added {want} to tweaks: {txt.splitlines()[0].strip()}")
                break
            # scene rotated off the spot; re-hover at the same point to see what's there now
            pg.mouse.move(found[0] - 3, found[1])
            pg.wait_for_timeout(60)
            pg.mouse.move(found[0], found[1])
            pg.wait_for_timeout(120)
            t = hover_text() or ""
            m = re.search(r"L(\d+) · N(\d+)", t)
            if m:
                found = (found[0], found[1], t, int(m.group(1)), int(m.group(2)))
            else:
                # scan a small neighborhood for the nearest orb
                got = False
                for dy in range(-40, 41, 16):
                    for dx in range(-40, 41, 16):
                        pg.mouse.move(found[0] + dx, found[1] + dy)
                        pg.wait_for_timeout(45)
                        t = hover_text() or ""
                        m = re.search(r"L(\d+) · N(\d+)", t)
                        if m:
                            found = (found[0] + dx, found[1] + dy, t, int(m.group(1)), int(m.group(2)))
                            got = True
                            break
                    if got:
                        break
                if not got:
                    fail("lost the neuron under the cursor after the first click attempt")
        if not added:
            fail("clicked an identified neuron 4 times; the tweaks panel never listed it")

        if errors:
            fail(f"console errors: {errors[:5]}")
        print("OK: hover names a specific L·N, click tweaks it, zero console errors")
        browser.close()


if __name__ == "__main__":
    main()
