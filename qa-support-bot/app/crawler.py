"""Bounded, same-origin crawl of public HTML pages; for trusted CLI operators."""
from collections import deque
from dataclasses import dataclass
from urllib.parse import urljoin, urlsplit, urlunsplit
from urllib.robotparser import RobotFileParser
import logging
import time
import httpx
from bs4 import BeautifulSoup

LOG = logging.getLogger(__name__)
AGENT = 'SupportBot/1.0'

@dataclass(frozen=True)
class Page:
    url: str
    text: str


def canonical(url: str) -> str:
    p = urlsplit(url)
    if p.scheme not in ('http', 'https') or not p.hostname or p.username or p.password:
        raise ValueError('Expected an HTTP(S) URL without credentials')
    return urlunsplit((p.scheme, p.netloc.lower(), p.path or '/', p.query, ''))


def origin(url: str) -> tuple:
    p = urlsplit(url)
    return p.scheme, p.hostname, p.port or (443 if p.scheme == 'https' else 80)


def clean_html(html: str) -> tuple[str, list[str]]:
    soup = BeautifulSoup(html, 'html.parser')
    links = [a['href'] for a in soup.find_all('a', href=True)]
    for tag in soup.select('script,style,noscript,nav,header,footer,aside,form,svg,[hidden]'):
        tag.decompose()
    body = soup.find('main') or soup.find('article') or soup.body or soup
    return ' '.join(body.get_text(' ', strip=True).split()), links


def crawl(start: str, client: httpx.Client, max_pages: int = 30, delay: float = 0.25):
    start = canonical(start)
    if max_pages < 1 or delay < 0:
        raise ValueError('Invalid crawl limits')
    base = origin(start)
    robots = RobotFileParser()
    robots_url = urljoin(start, '/robots.txt')
    response = client.get(robots_url, headers={'User-Agent': AGENT}, follow_redirects=False)
    if response.status_code in (404, 410):
        robots.parse([])
    elif response.status_code == 200:
        robots.parse(response.text.splitlines())
    else:
        raise RuntimeError(f'Cannot safely read robots.txt: HTTP {response.status_code}')
    delay = max(delay, robots.crawl_delay(AGENT) or 0)
    pending, seen = deque([start]), {start}
    attempted = 0
    while pending and attempted < max_pages:
        url = pending.popleft()
        if not robots.can_fetch(AGENT, url):
            continue
        attempted += 1
        time.sleep(delay)
        try:
            with client.stream('GET', url, headers={'User-Agent': AGENT}, follow_redirects=False) as res:
                if res.is_redirect:
                    links = [res.headers.get('location', '')]
                    text = ''
                elif res.status_code == 200 and 'text/html' in res.headers.get('content-type', '').lower():
                    data = bytearray()
                    for part in res.iter_bytes():
                        data.extend(part)
                        if len(data) > 2_000_000:
                            break
                    if len(data) > 2_000_000:
                        LOG.warning('Page too large: %s', url)
                        continue
                    text, links = clean_html(data.decode(res.encoding or 'utf-8', errors='replace'))
                else:
                    LOG.warning('Skipping %s (HTTP %s or non-HTML)', url, res.status_code)
                    continue
        except httpx.HTTPError as exc:
            LOG.warning('Fetch failed for %s: %s', url, type(exc).__name__)
            continue
        if text:
            yield Page(url, text)
        for href in links:
            try:
                target = canonical(urljoin(url, href))
                if origin(target) == base and target not in seen and len(seen) < max_pages * 20:
                    seen.add(target)
                    pending.append(target)
            except ValueError:
                continue
