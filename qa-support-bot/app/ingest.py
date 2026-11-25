"""CLI: crawl, chunk, embed, and index one website into a new collection."""
import argparse
import logging
from itertools import islice
import httpx
from .crawler import crawl
from .chunking import chunk_page
from .providers import OpenAI, Qdrant


def ingest(pages, embedder, store, batch_size=16):
    chunks = (chunk for page in pages for chunk in chunk_page(page))
    created, count = False, 0
    try:
        while batch := list(islice(chunks, batch_size)):
            vectors = embedder.embed([c.text for c in batch])
            if not vectors or len(vectors) != len(batch) or not vectors[0] or any(len(v) != len(vectors[0]) for v in vectors):
                raise ValueError('Invalid embedding batch')
            if not created:
                store.create(len(vectors[0]))
                created = True
            store.upsert(batch, vectors)
            count += len(batch)
        if not count:
            raise ValueError('No crawlable text found')
        return count
    except Exception:
        if created:
            try:
                store.delete()
            except Exception:
                logging.exception('Could not clean up incomplete collection; remove it manually')
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('url', help='Public HTTP(S) website URL (trusted operator input)')
    parser.add_argument('--max-pages', type=int, default=30)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format='%(levelname)s %(message)s')
    with httpx.Client(timeout=45) as provider_client, httpx.Client(timeout=15) as crawl_client:
        ai, store = OpenAI(provider_client), Qdrant(provider_client)
        # Do not overwrite a previous collection, including when Qdrant accepts a repeat create.
        existing = provider_client.get(store.url + store.path, headers=store.headers)
        if existing.status_code != 404:
            existing.raise_for_status()
            raise SystemExit('Collection already exists. Set QDRANT_COLLECTION to a new name.')
        count = ingest(crawl(args.url, crawl_client, args.max_pages), ai, store)
        print(f'Indexed {count} chunks into {store.collection}')

if __name__ == '__main__':
    main()
