"""Compare spreadsheet product UPCs with public InComm image directories."""
from concurrent.futures import ThreadPoolExecutor
from decimal import Decimal, InvalidOperation
from html.parser import HTMLParser
from http.client import HTTPException
import io
import posixpath
import re
from urllib.error import HTTPError, URLError
from urllib.parse import unquote, urljoin, urlsplit, urlunsplit
from urllib.request import Request, urlopen, HTTPRedirectHandler, build_opener

import openpyxl
from app.holiday_planograms.normalization import upc_a_check_digit


DEFAULT_IMAGE_URL = 'https://assets.incomm.com/Images/'
IMAGE_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.webp', '.gif', '.tif', '.tiff')


def _upc_digits(value):
    if value is None or isinstance(value, bool):
        return None
    text = str(value).strip()
    try:
        if not re.fullmatch(r'\d+', text):
            number = Decimal(text)
            if not number.is_finite() or number != number.to_integral_value():
                return None
            text = str(int(number))
    except InvalidOperation:
        return None
    return text if 8 <= len(text) <= 14 and int(text) > 0 else None


def normalize_upc(value):
    digits = _upc_digits(value)
    
    if not digits:
        return None
    
    digits = digits.lstrip("0")
    return digits[:11] or None
    


def comparison_upc(value, bos_upc11):
    return normalize_upc(value)

def read_comparison_workbook(data):
    if len(data) > 25 * 1024 * 1024:
        raise ValueError('Upload an Excel workbook smaller than 25 MB.')
    try:
        workbook = openpyxl.load_workbook(io.BytesIO(data), read_only=True, data_only=True)
    except Exception as exc:
        raise ValueError('Unable to read this Excel workbook. Upload a valid .xlsx file.') from exc
    sheets = {}
    try:
        for sheet in workbook:
            mapping = None
            products = []
            for number, values in enumerate(sheet.iter_rows(values_only=True), 1):
                if mapping is None:
                    if number > 100:
                        break
                    labels = {re.sub(r'[^a-z0-9]', '', str(v).lower()): index
                              for index, v in enumerate(values) if v is not None}
                    upc = next((labels[k] for k in ('upc', 'productupc', 'packupc') if k in labels), None)
                    if upc is None:
                        continue
                    mapping = {'upc':upc}
                    for field, aliases in [('name', ('name','productname','description')),
                                            ('item', ('id','itemnumber','wmitemnumber','item'))]:
                        mapping[field] = next((labels[k] for k in aliases if k in labels), None)
                    continue
                record = {key: values[index] if index is not None and index < len(values) else None
                          for key, index in mapping.items()}
                if not any(v is not None and str(v).strip() for v in record.values()):
                    continue
                if all(str(v or '').strip().lower() in ('upc','name','id','item number','')
                       for v in record.values()):
                    continue
                products.append({'Workbook row':number, 'UPC':str(record['upc'] or '').strip(),
                                 'Product':str(record['name'] or '').strip(),
                                 'Item #':str(record['item'] or '').strip()})
            if mapping and products:
                sheets[sheet.title] = products
    finally:
        workbook.close()
    if not sheets:
        raise ValueError('No product table with a UPC column was found. Use a sheet with UPC and product rows.')
    return sheets


def validate_image_url(url):
    parts = urlsplit(url.strip())
    if (parts.scheme != 'https' or parts.hostname != 'assets.incomm.com' or
            parts.username or parts.password or parts.port not in (None, 443) or
            parts.query or parts.fragment):
        raise ValueError('Enter an HTTPS InComm image-directory link, such as https://assets.incomm.com/Images/.')
    path = posixpath.normpath(unquote(parts.path))
    if path != '/Images' and not path.startswith('/Images/'):
        raise ValueError('Use a folder within the InComm /Images/ directory.')
    return urlunsplit(('https', 'assets.incomm.com', path.rstrip('/') + '/', '', ''))


class _SafeRedirects(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        validate_image_url(newurl)
        return super().redirect_request(req, fp, code, msg, headers, newurl)


class _Links(HTMLParser):
    def __init__(self):
        super().__init__()
        self.hrefs = []

    def handle_starttag(self, tag, attrs):
        if tag.lower() == 'a':
            self.hrefs.extend(value for key, value in attrs if key == 'href' and value)


class ImageDirectory:
    def __init__(self):
        self.opener = build_opener(_SafeRedirects())

    def links(self, url):
        request = Request(url, headers={'User-Agent':'Planogram-UPC-Comparison/1.0'})
        try:
            with self.opener.open(request, timeout=15) as response:
                data = response.read(5 * 1024 * 1024 + 1)
                if len(data) > 5 * 1024 * 1024:
                    raise ValueError('The image-directory listing is too large to compare.')
                parser = _Links()
                parser.feed(data.decode('utf-8', errors='replace'))
        except (HTTPError, URLError, TimeoutError, OSError, HTTPException) as exc:
            raise ValueError(f'Could not read the image directory: {url}') from exc
        links = []
        for href in parser.hrefs:
            target = urljoin(url, href)
            parts = urlsplit(target)
            # Ignore sort links, parents, other hosts, and non-descendant paths.
            if (parts.scheme == 'https' and parts.netloc == 'assets.incomm.com' and
                    not parts.query and not parts.fragment and parts.path.startswith(urlsplit(url).path) and
                    target != url):
                path = posixpath.normpath(unquote(parts.path))
                if path.startswith(urlsplit(url).path):
                    links.append(target)
        return sorted(set(links))

    def image_exists(self, url):
        request = Request(url, method='HEAD', headers={'User-Agent':'Planogram-UPC-Comparison/1.0'})
        try:
            with self.opener.open(request, timeout=15) as response:
                return response.headers.get_content_type().startswith('image/')
        except HTTPError as exc:
            if exc.code in (404, 410):
                return False
            raise ValueError('The matching image could not be verified.') from exc
        except (URLError, TimeoutError, OSError, HTTPException) as exc:
            raise ValueError('The matching image could not be verified.') from exc


def compare_products(products, url, directory=None, *, bos_upc11=True):
    root = validate_image_url(url)
    if not products:
        raise ValueError('The selected sheet has no product rows to compare.')
    directory = directory or ImageDirectory()
    links = directory.links(root)
    if root.endswith('/Images/'):
        categories = [link for link in links if link.rstrip('/').rsplit('/',1)[-1] in ('Product','Pack')]
        if not categories:
            raise ValueError('The image library did not expose Product or Pack directories. No score was calculated.')
        links = []
        for category in categories:
            links.extend(directory.links(category))
    folders, files = {}, {}
    for link in links:
        name = unquote(urlsplit(link).path.rstrip('/').rsplit('/',1)[-1])
        if link.endswith('/'):
            key = normalize_upc(name)
            if key:
                folders.setdefault(key, []).append((name, link))
        elif name.lower().endswith(IMAGE_EXTENSIONS):
            for number in re.findall(r'(?<!\d)\d{8,14}(?!\d)', name):
                key = normalize_upc(number)
                if key:
                    files.setdefault(key, []).append((number, link))

    def resolve(key):
        if key is None:
            return 'Missing / invalid UPC', '', ''
        candidates = list(files.get(key, []))
        try:
            for folder_upc, folder in folders.get(key, []):
                for link in directory.links(folder):
                    if unquote(urlsplit(link).path).lower().endswith(IMAGE_EXTENSIONS):
                        candidates.append((folder_upc, link))
            for matched_upc, image in candidates:
                if directory.image_exists(image):
                    return 'Matched', matched_upc, image
        except ValueError:
            return 'Unverified - connection or access error', '', ''
        return 'No matching image', '', ''

    keys = list(dict.fromkeys(normalize_upc(comparison_upc(row['UPC'], bos_upc11)) for row in products))
    with ThreadPoolExecutor(max_workers=6) as pool:
        matches = dict(zip(keys, pool.map(resolve, keys)))
    rows = []
    for product in products:
        compared = comparison_upc(product['UPC'], bos_upc11)
        status, upc, image = matches[normalize_upc(compared)]
        rows.append({**product, 'Comparison UPC':compared or '',
                     'Status':status, 'Matched UPC':upc, 'Image':image})
    matched = sum(row['Status'] == 'Matched' for row in rows)
    unverified = sum(row['Status'].startswith('Unverified') for row in rows)
    return {'rows':rows, 'total':len(rows), 'matched':matched, 'unverified':unverified,
            'missing':len(rows)-matched-unverified,
            'accuracy':100 * matched / len(rows) if not unverified else None}
