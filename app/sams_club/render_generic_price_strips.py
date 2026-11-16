"""Generic dimensions use the same native Gibson ticket artwork as Holiday."""


def render_generic_price_strips_pdf(rows, *, calibration=False):
    from app.sams_club.render_price_strips_html import render_sams_price_strips_pdf

    return render_sams_price_strips_pdf(rows, template_name='Generic', calibration=calibration)
