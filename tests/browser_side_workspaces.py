"""Live 4F Chromium stress check. Start Streamlit first; see docs/direct_canvas_builder.md."""
import argparse
from playwright.sync_api import expect, sync_playwright


def check(url, screenshot=None):
    expect.set_options(timeout=15000)
    with sync_playwright() as p:
        browser=p.chromium.launch(headless=True)
        page=browser.new_page(viewport={'width':1600,'height':1100})
        page.goto(url)
        page.get_by_test_id('stSelectbox').first.click()
        page.get_by_role('option',name='Templates',exact=True).click()
        for name in ('Create New Template','Start blank draft','Add Side'):
            page.get_by_role('button',name=name,exact=True).click()
        cards=page.locator('[class*="st-key-side_workspace"]')
        source=cards.nth(0);frame=source.frame_locator('iframe')
        expected_boxes=0
        def add_section(name,rows=None,columns=None):
            nonlocal expected_boxes
            source.get_by_text('Add a section',exact=True).click()
            page.get_by_role('textbox',name='New section name',exact=True).fill(name)
            page.get_by_role('button',name='Add Section',exact=True).click()
            expect(source.get_by_role('button',name=name,exact=True)).to_be_visible()
            page.keyboard.press('Escape')
            if rows:
                source.get_by_role('spinbutton',name='Rows',exact=True).fill(str(rows))
                source.get_by_role('spinbutton',name='Columns',exact=True).fill(str(columns))
                source.get_by_role('button',name='Create Grid',exact=True).click()
                expected_boxes+=rows*columns
                expect(frame.locator('[data-slot-id]')).to_have_count(expected_boxes)
            else:
                for _ in range(3):
                    source.get_by_role('button',name='Add Box',exact=True).click()
                    expected_boxes+=1
                    expect(frame.locator('[data-slot-id]')).to_have_count(expected_boxes)
        add_section('Top Cards',2,5)
        add_section('Main Cards',4,8)
        add_section('Holders',1,5)
        add_section('Bonus')
        expect(frame.locator('[data-slot-id]')).to_have_count(50)
        expect(frame.locator('.handle')).to_have_count(0)
        for name in ('Wider','Narrower','Taller','Shorter'):
            expect(page.get_by_role('button',name=name,exact=True)).to_have_count(0)
        def scene(f):
            return f.locator('[data-slot-id]').evaluate_all('els=>els.map(e=>Object.fromEntries(["data-slot-id","x","y","width","height"].map(k=>[k,e.getAttribute(k)])))')
        first_scene=scene(frame)
        assert len({(round(float(s['width']),6),round(float(s['height']),6)) for s in first_scene})==1
        def settle(f):expect(f.locator('#status')).not_to_contain_text('Updating')
        def select_section(name):
            source.get_by_role('button',name=name,exact=True).click()
            rid=frame.locator('[data-region-label]').filter(has_text=name).get_attribute('data-region-label')
            expect(frame.locator('[data-region-id].active-section')).to_have_attribute('data-region-id',rid)
        def drag(f,box,dx,dy):
            box.scroll_into_view_if_needed();r=box.bounding_box();x=r['x']+r['width']/2;y=r['y']+r['height']/2
            page.mouse.move(x,y);page.mouse.down();page.mouse.move(x+dx,y+dy,steps=8);page.mouse.up();settle(f)
        # Manually arrange a Bonus box, preserving its fixed dimensions.
        bonus=frame.locator('[data-slot-id]').nth(47)
        bonus.click();settle(frame);drag(frame,bonus,5,6)
        arranged=scene(frame)
        assert arranged[47]['x']!=first_scene[47]['x']
        assert (arranged[47]['width'],arranged[47]['height'])==(first_scene[47]['width'],first_scene[47]['height'])
        select_section('Top Cards')
        source.get_by_text('Duplicate Section',exact=True).click()
        page.get_by_role('button',name='Create Section Copy',exact=True).click();page.keyboard.press('Escape')
        expect(frame.locator('[data-slot-id]')).to_have_count(60)
        expect(source.get_by_role('button',name='Top Cards Copy',exact=True)).to_be_visible()
        source.get_by_text('Delete Section',exact=True).click()
        # Confirmation is required; opening the action does not remove anything.
        expect(frame.locator('[data-slot-id]')).to_have_count(60)
        page.get_by_role('button',name='Confirm Delete Section',exact=True).click();page.keyboard.press('Escape')
        expect(frame.locator('[data-slot-id]')).to_have_count(50)
        # Select and move an entire 32-box Section to exercise more than a few cells.
        select_section('Main Cards')
        source.locator('iframe').scroll_into_view_if_needed()
        label=frame.locator('[data-region-label]').filter(has_text='Main Cards')
        region_id=label.get_attribute('data-region-label')
        frame.locator(f'[data-region-id="{region_id}"]').scroll_into_view_if_needed()
        frame.get_by_role('button',name='Select boxes',exact=True).click()
        many=frame.locator('[data-slot-id]')
        a=many.nth(10).bounding_box();b=many.nth(41).bounding_box()
        page.mouse.move(a['x']-4,a['y']-4);page.mouse.down()
        page.mouse.move(b['x']+b['width']+4,b['y']+b['height']+4,steps=25);page.mouse.up();settle(frame)
        expect(frame.locator('.selected')).to_have_count(32)
        frame.get_by_role('button',name='Move boxes',exact=True).click()
        before_many=scene(frame);drag(frame,many.nth(10),2,2);after_many=scene(frame)
        assert after_many[:10]==before_many[:10] and after_many[42:]==before_many[42:]
        assert all(a['x']!=b['x'] and a['y']!=b['y'] for a,b in zip(before_many[10:42],after_many[10:42]))
        select_section('Top Cards')
        frame.locator('#canvas').evaluate('e=>{e.scrollTop=0;e.scrollLeft=0}')
        frame.get_by_role('button',name='Select boxes',exact=True).click()
        boxes=frame.locator('[data-slot-id]')
        boxes.first.scroll_into_view_if_needed()
        a=boxes.nth(0).bounding_box();b=boxes.nth(2).bounding_box()
        page.mouse.move(a['x']-4,a['y']-4);page.mouse.down()
        page.mouse.move(b['x']+b['width']+4,b['y']+b['height']+4,steps=15);page.mouse.up();settle(frame)
        expect(frame.locator('.selected')).to_have_count(3)
        frame.get_by_role('button',name='Move boxes',exact=True).click()
        before_group=scene(frame);drag(frame,boxes.first,4,4);after_group=scene(frame)
        for a,b in zip(before_group[:3],after_group[:3]):
            assert a['x']!=b['x'] and a['y']!=b['y']
            assert (a['data-slot-id'],a['width'],a['height'])==(b['data-slot-id'],b['width'],b['height'])
        assert after_group[3:]==before_group[3:]
        source.get_by_role('button',name='Delete Selected',exact=True).click()
        expect(boxes).to_have_count(47);expect(frame.locator('.selected')).to_have_count(0)
        preserved=scene(frame)
        source.get_by_text('Duplicate Side',exact=True).first.click()
        source.get_by_role('textbox',name='Copy side name',exact=True).fill('Side B')
        source.get_by_role('button',name='Duplicate Side',exact=True).click()
        expect(cards).to_have_count(2);copy=cards.nth(1);copy_frame=copy.frame_locator('iframe')
        expect(copy_frame.locator('[data-slot-id]')).to_have_count(47)
        copy_scene=scene(copy_frame)
        assert not ({s['data-slot-id'] for s in preserved}&{s['data-slot-id'] for s in copy_scene})
        assert [{k:v for k,v in s.items() if k!='data-slot-id'} for s in preserved]==[{k:v for k,v in s.items() if k!='data-slot-id'} for s in copy_scene]
        copy_frame.locator('[data-slot-id]').first.click();settle(copy_frame)
        copy.get_by_role('button',name='Delete Selected',exact=True).click()
        expect(copy_frame.locator('[data-slot-id]')).to_have_count(46)
        assert scene(frame)==preserved
        expect(page.get_by_test_id('stException')).to_have_count(0)
        for iframe in page.locator('iframe').all():assert int(iframe.get_attribute('height'))<=760
        if screenshot:
            source.get_by_role('heading',name='Side 1',exact=True).scroll_into_view_if_needed()
            page.screenshot(path=screenshot)
        page.set_viewport_size({'width':1100,'height':900})
        assert frame.locator('#canvas').evaluate('e=>e.clientHeight')<=640
        browser.close()
    print('PASS: 50-box/four-Section irregular side, fixed sizes, no resize, Section duplication/confirmed deletion, rectangle selection, group drag, bulk delete, independent Side copy and contained responsive canvases.')


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--url',default='http://127.0.0.1:8514');parser.add_argument('--screenshot')
    args=parser.parse_args();check(args.url,args.screenshot)
