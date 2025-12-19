#!/usr/bin/env python3
"""
BibTeX to Google Scholar Link Extractor

This script parses multiple .bib files, searches each entry's title on Google Scholar,
and exports the results (including the first article link) to a CSV file.

Usage:
    python bib_to_scholar_links.py file1.bib file2.bib ... -o output.csv

Requirements:
    pip install bibtexparser scholarly pandas
"""

import argparse
import csv
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

try:
    import bibtexparser
    import pandas as pd
    from scholarly import scholarly, ProxyGenerator
except ImportError as e:
    print(f"Error: Missing required package. Please install dependencies:")
    print(f"  pip install bibtexparser scholarly pandas")
    sys.exit(1)


def normalize_author_name(name: str) -> str:
    """
    Normalize author name for comparison.

    Removes special characters, converts to lowercase, handles different formats.

    Args:
        name: Author name in various formats

    Returns:
        Normalized name (last name only, lowercase)
    """
    # Remove LaTeX brackets and special characters
    name = name.strip('{}').replace('{', '').replace('}', '')

    # Split by 'and' to get first author
    if ' and ' in name.lower():
        name = name.split(' and ')[0].strip()

    # Handle "Last, First" format
    if ',' in name:
        parts = name.split(',')
        last_name = parts[0].strip()
    else:
        # Handle "First Last" or "First Middle Last" format
        parts = name.split()
        last_name = parts[-1] if parts else name

    # Normalize: lowercase, remove special chars
    last_name = last_name.lower().strip()
    last_name = ''.join(c for c in last_name if c.isalnum())

    return last_name


def extract_first_author(bib_entry: Dict) -> Optional[str]:
    """
    Extract and normalize the first author from a BibTeX entry.

    Args:
        bib_entry: BibTeX entry dictionary

    Returns:
        Normalized first author last name, or None if not found
    """
    author_field = bib_entry.get('author', '')
    if not author_field:
        return None

    return normalize_author_name(author_field)


def parse_bib_files(bib_files: List[str]) -> List[Dict]:
    """
    Parse multiple BibTeX files and extract entries.

    Args:
        bib_files: List of paths to .bib files

    Returns:
        List of dictionaries containing BibTeX entry data
    """
    all_entries = []

    for bib_file in bib_files:
        try:
            with open(bib_file, 'r', encoding='utf-8') as f:
                bib_database = bibtexparser.load(f)
                for entry in bib_database.entries:
                    entry['source_file'] = bib_file
                    all_entries.append(entry)
            print(f"✓ Parsed {len(bib_database.entries)} entries from {bib_file}")
        except FileNotFoundError:
            print(f"✗ Error: File not found: {bib_file}")
        except Exception as e:
            print(f"✗ Error parsing {bib_file}: {e}")

    return all_entries


def search_google_scholar(
    title: str,
    expected_author: Optional[str] = None,
    max_retries: int = 3
) -> tuple[Optional[str], str]:
    """
    Search Google Scholar for a title and return the first result's link.

    Args:
        title: Paper title to search
        expected_author: Expected first author last name (normalized)
        max_retries: Maximum number of retry attempts

    Returns:
        Tuple of (URL or None, match_status)
        match_status: 'MATCH', 'MISMATCH', 'NO_AUTHOR', 'NOT_FOUND', 'ERROR'
    """
    for attempt in range(max_retries):
        try:
            # Search for the title
            search_query = scholarly.search_pubs(title)

            # Get first result
            first_result = next(search_query, None)

            if first_result:
                # Try to get the URL from different fields
                url = (
                    first_result.get('pub_url') or
                    first_result.get('eprint_url') or
                    first_result.get('url_scholarbib')
                )

                # Verify author if expected_author is provided
                if expected_author:
                    # Get author from result
                    scholar_author = first_result.get('bib', {}).get('author', [''])[0]
                    if scholar_author:
                        scholar_author_normalized = normalize_author_name(scholar_author)

                        if scholar_author_normalized == expected_author:
                            return url, 'MATCH'
                        else:
                            return url, f'MISMATCH (found: {scholar_author})'
                    else:
                        return url, 'NO_AUTHOR'
                else:
                    return url, 'NO_AUTHOR_CHECK'

            return None, 'NOT_FOUND'

        except StopIteration:
            return None, 'NOT_FOUND'
        except Exception as e:
            if attempt < max_retries - 1:
                wait_time = (attempt + 1) * 5  # Exponential backoff
                print(f"  ⚠ Retry {attempt + 1}/{max_retries} after {wait_time}s: {str(e)[:50]}")
                time.sleep(wait_time)
            else:
                print(f"  ✗ Failed after {max_retries} attempts: {str(e)[:50]}")
                return None, f'ERROR: {str(e)[:50]}'

    return None, 'ERROR'


def process_entries(entries: List[Dict], delay: float = 2.0) -> List[Dict]:
    """
    Process BibTeX entries and search Google Scholar for each.

    Args:
        entries: List of BibTeX entry dictionaries
        delay: Delay in seconds between requests (to avoid rate limiting)

    Returns:
        List of dictionaries with added 'scholar_url' and 'author_match' fields
    """
    results = []
    total = len(entries)

    print(f"\n🔍 Searching Google Scholar for {total} entries...")
    print(f"⏱  Estimated time: ~{int(total * delay / 60)} minutes\n")

    for i, entry in enumerate(entries, 1):
        title = entry.get('title', '').strip('{}')
        cite_key = entry.get('ID', 'unknown')

        if not title:
            print(f"[{i}/{total}] ⚠ Skipping entry '{cite_key}': No title found")
            entry['scholar_url'] = 'NO_TITLE'
            entry['author_match'] = 'NO_TITLE'
            results.append(entry)
            continue

        # Extract expected author
        expected_author = extract_first_author(entry)

        print(f"[{i}/{total}] Searching: {title[:60]}...")
        if expected_author:
            print(f"  Expected author: {expected_author}")

        url, match_status = search_google_scholar(title, expected_author)

        if url:
            print(f"  ✓ Found: {url}")
            print(f"  Author match: {match_status}")
            entry['scholar_url'] = url
            entry['author_match'] = match_status
        else:
            print(f"  ✗ Not found (status: {match_status})")
            entry['scholar_url'] = match_status
            entry['author_match'] = match_status

        results.append(entry)

        # Delay to avoid rate limiting (except for last entry)
        if i < total:
            time.sleep(delay)

    return results


def export_to_csv(entries: List[Dict], output_file: str):
    """
    Export results to CSV file.

    Args:
        entries: List of BibTeX entries with scholar_url and author_match fields
        output_file: Path to output CSV file
    """
    if not entries:
        print("No entries to export.")
        return

    # Select fields to export
    fields = ['ID', 'title', 'author', 'year', 'journal', 'booktitle',
              'scholar_url', 'author_match', 'source_file', 'ENTRYTYPE']

    # Prepare data
    rows = []
    for entry in entries:
        row = {}
        for field in fields:
            value = entry.get(field, '')
            # Clean up LaTeX brackets
            if isinstance(value, str):
                value = value.strip('{}')
            row[field] = value
        rows.append(row)

    # Export to CSV
    df = pd.DataFrame(rows)
    df.to_csv(output_file, index=False, encoding='utf-8')

    print(f"\n✓ Exported {len(rows)} entries to {output_file}")

    # Print summary statistics
    found_with_match = sum(1 for e in entries if e.get('author_match') == 'MATCH')
    found_with_mismatch = sum(1 for e in entries if 'MISMATCH' in e.get('author_match', ''))
    found_no_author = sum(1 for e in entries if e.get('author_match') in ['NO_AUTHOR', 'NO_AUTHOR_CHECK'])
    not_found = sum(1 for e in entries if e.get('author_match') == 'NOT_FOUND')
    no_title = sum(1 for e in entries if e.get('author_match') == 'NO_TITLE')
    errors = sum(1 for e in entries if 'ERROR' in e.get('author_match', ''))

    print(f"\n📊 Summary:")
    print(f"  Author match:      {found_with_match}/{len(entries)} ({found_with_match/len(entries)*100:.1f}%)")
    print(f"  Author mismatch:   {found_with_mismatch}/{len(entries)} ({found_with_mismatch/len(entries)*100:.1f}%)")
    print(f"  Found (no author): {found_no_author}/{len(entries)} ({found_no_author/len(entries)*100:.1f}%)")
    print(f"  Not found:         {not_found}/{len(entries)} ({not_found/len(entries)*100:.1f}%)")
    print(f"  No title:          {no_title}/{len(entries)} ({no_title/len(entries)*100:.1f}%)")
    if errors > 0:
        print(f"  Errors:            {errors}/{len(entries)} ({errors/len(entries)*100:.1f}%)")


def setup_proxy(use_proxy: bool = True):
    """
    Setup ProxyGenerator for scholarly to avoid rate limiting.

    Args:
        use_proxy: Whether to use free proxies
    """
    if use_proxy:
        print("\n🔧 Setting up proxy...")
        try:
            pg = ProxyGenerator()
            success = pg.FreeProxies()
            if success:
                scholarly.use_proxy(pg)
                print("✓ Proxy configured successfully")
            else:
                print("⚠ Proxy setup failed, continuing without proxy")
        except Exception as e:
            print(f"⚠ Proxy setup error: {e}")
            print("  Continuing without proxy...")
    else:
        print("\n⚠ Running without proxy (may hit rate limits)")


def main():
    parser = argparse.ArgumentParser(
        description='Parse BibTeX files and search Google Scholar for article links',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Examples:
  # Process single file
  python bib_to_scholar_links.py references.bib -o output.csv

  # Process multiple files
  python bib_to_scholar_links.py refs1.bib refs2.bib refs3.bib -o output.csv

  # Adjust delay between requests
  python bib_to_scholar_links.py *.bib -o output.csv --delay 3.0

  # Run without proxy (faster but may hit rate limits)
  python bib_to_scholar_links.py *.bib -o output.csv --no-proxy
        '''
    )

    parser.add_argument(
        'bib_files',
        nargs='+',
        help='One or more .bib files to process'
    )

    parser.add_argument(
        '-o', '--output',
        default='scholar_links.csv',
        help='Output CSV file (default: scholar_links.csv)'
    )

    parser.add_argument(
        '--delay',
        type=float,
        default=2.0,
        help='Delay in seconds between Google Scholar requests (default: 2.0)'
    )

    parser.add_argument(
        '--no-proxy',
        action='store_true',
        help='Disable proxy usage (faster but may hit rate limits)'
    )

    args = parser.parse_args()

    print("=" * 70)
    print("BibTeX to Google Scholar Link Extractor")
    print("=" * 70)

    # Setup proxy
    setup_proxy(use_proxy=not args.no_proxy)

    # Parse BibTeX files
    entries = parse_bib_files(args.bib_files)

    if not entries:
        print("\n✗ No entries found in the provided files.")
        sys.exit(1)

    # Search Google Scholar
    results = process_entries(entries, delay=args.delay)

    # Export to CSV
    export_to_csv(results, args.output)

    print("\n✓ Done!")


if __name__ == '__main__':
    main()
