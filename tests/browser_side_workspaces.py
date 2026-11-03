"""Manual live Chromium check. Start Streamlit, then run this script (see docs)."""
import argparse

from playwright.sync_api import expect, sync_playwright


def check(url, screenshot=None):
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page(viewport={'width':1400, 'height':1000})
        page.goto(url)
        page.get_by_test_id('stSelectbox').first.click()
        page.get_by_role('option',name='Templates',exact=True).click()
        for label in ('Create New Template','Start blank draft','Add Side'):
            page.get_by_role('button',name=label,exact=True).click()
        page.get_by_text('Add a section',exact=True).click()
        page.get_by_role('button',name='Add Section',exact=True).click()
        page.keyboard.press('Escape')
        page.get_by_role('spinbutton',name='Rows',exact=True).fill('3')
        page.get_by_role('spinbutton',name='Columns',exact=True).fill('4')
        page.get_by_role('button',name='Create Grid',exact=True).click()
        cards=page.locator('[class*="st-key-side_workspace"]')
        original=cards.nth(0)
        original.get_by_text('Duplicate Side',exact=True).first.click()
        original.get_by_role('textbox',name='Copy side name',exact=True).fill('Side B')
        original.get_by_role('button',name='Duplicate Side',exact=True).click()
        expect(cards).to_have_count(2)
        copy=cards.nth(1)
        source_frame=original.frame_locator('iframe')
        copy_frame=copy.frame_locator('iframe')
        expect(source_frame.locator('[data-slot-id]')).to_have_count(12)
        expect(copy_frame.locator('[data-slot-id]')).to_have_count(12)

        def scene(frame):
            return frame.locator('[data-slot-id]').evaluate_all('els => els.map(e => Object.fromEntries(["data-slot-id","x","y","width","height"].map(k => [k,e.getAttribute(k)])))')

        source_before=scene(source_frame)
        copy_before=scene(copy_frame)
        assert not ({b['data-slot-id'] for b in source_before} & {b['data-slot-id'] for b in copy_before})
        for a,b in zip(source_before,copy_before):
            assert {k:v for k,v in a.items() if k!='data-slot-id'}=={k:v for k,v in b.items() if k!='data-slot-id'}

        def gesture(frame, target, dx, dy):
            target.scroll_into_view_if_needed()
            r=target.bounding_box();x=r['x']+r['width']/2;y=r['y']+r['height']/2
            page.mouse.move(x,y);page.mouse.down();page.mouse.move(x+dx,y+dy,steps=6);page.mouse.up()
            expect(frame.locator('#status')).not_to_contain_text('Updating')

        box=copy_frame.locator('[data-slot-id]').first
        box.click()
        expect(copy_frame.locator('#status')).not_to_contain_text('Updating')
        expect(copy_frame.locator('.handle')).to_have_count(8)
        gesture(copy_frame,box,9,11)
        moved=scene(copy_frame)
        assert moved[0]['x']!=copy_before[0]['x'] and moved[0]['y']!=copy_before[0]['y']
        assert moved[1:]==copy_before[1:]
        assert scene(source_frame)==source_before
        gesture(copy_frame,copy_frame.locator('[data-handle="se"]'),35,28)
        resized=scene(copy_frame)
        assert resized[0]['width']!=moved[0]['width']
        assert resized[0]['data-slot-id']==copy_before[0]['data-slot-id']
        assert resized[1:]==copy_before[1:]
        assert scene(source_frame)==source_before
        expect(copy.get_by_test_id('stAlert').filter(has_text='overlap')).to_have_count(1)
        expect(copy_frame.locator('[data-slot-id]')).to_have_count(12)
        # Switch back to another Side and edit its canvas after the copy changed.
        source_box=source_frame.locator('[data-slot-id]').first
        source_box.click()
        expect(source_frame.locator('#status')).not_to_contain_text('Updating')
        gesture(source_frame,source_box,4,5)
        assert scene(copy_frame)==resized
        source_after=scene(source_frame)
        copy.get_by_role('button',name='Delete Box',exact=True).click()
        expect(copy_frame.locator('[data-slot-id]')).to_have_count(11)
        assert scene(source_frame)==source_after
        original.get_by_text('Add a section',exact=True).click()
        # Popover contents are portaled outside the card's DOM subtree.
        page.get_by_role('textbox',name='New section name',exact=True).fill('Bonus')
        page.get_by_role('button',name='Add Section',exact=True).click()
        page.keyboard.press('Escape')
        expect(original.get_by_role('button',name='Bonus',exact=True)).to_be_visible()
        expect(source_frame.locator('.selected')).to_have_count(0)
        expect(copy.get_by_role('button',name='Bonus',exact=True)).to_have_count(0)
        expect(page.get_by_test_id('stException')).to_have_count(0)
        for iframe in page.locator('iframe').all():
            assert int(iframe.get_attribute('height')) <= 600
        if screenshot:
            original.scroll_into_view_if_needed()
            page.screenshot(path=screenshot)
        page.set_viewport_size({'width':1000,'height':850})
        expect(source_frame.locator('[data-slot-id]')).to_have_count(12)
        expect(copy_frame.locator('[data-slot-id]')).to_have_count(11)
        browser.close()
    print('PASS: two live Side canvases, independent IDs/geometry, selection, eight handles, drag, resize, overlap validation, local delete/Sections and compact sizing.')


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--url',default='http://127.0.0.1:8514')
    parser.add_argument('--screenshot')
    args=parser.parse_args()
    check(args.url,args.screenshot)
