"""
Parse Google Takeout Gemini MiActividad.html files.
Extracts ALL Gemini conversations from takeout and saves as JSONL for ingestion.
"""
import re
import json
import zipfile
from pathlib import Path
from datetime import datetime


def extract_from_gemini_takeout(zip_path):
    """Extract conversations from Gemini Google Takeout ZIP using streaming."""
    zip_path = Path(zip_path)
    
    all_entries = []
    
    with zipfile.ZipFile(zip_path, 'r') as zf:
        html_paths = find_gemini_html_in_zip(zf)
        
        for html_path in html_paths:
            info = zf.getinfo(html_path)
            print(f'Parsing {html_path} ({info.file_size/1e6:.1f} MB)...')
            
            entries = parse_gemini_html_stream(zf, html_path)
            print(f'  Found {len(entries)} conversations')
            all_entries.extend(entries)
            
            # Show samples
            for i, entry in enumerate(entries[:3]):
                prompt_preview = entry.get('prompt', '')[:80]
                resp_preview = entry.get('response', '')[:80]
                print(f'  #{i+1} [{entry.get("date", "?")}]')
                print(f'    Prompt: {prompt_preview}...')
                print(f'    Response: {resp_preview}...')
    
    print(f'\nTotal conversations extracted: {len(all_entries)}')
    return all_entries


def find_gemini_html_in_zip(zf):
    """Find MiActividad.html files for Gemini in a ZIP."""
    targets = []
    for name in zf.namelist():
        if ('gemini' in name.lower() or 'modo ia' in name.lower()) and name.endswith('.html'):
            targets.append(name)
    return targets


# Date pattern: "10 abr 2026, 19:55:59 CEST" (Google Takeout timestamp format)
DATE_PATTERN = re.compile(
    r'(\d{1,2}\s+\w{3,9}\s+\d{4},\s+\d{2}:\d{2}:\d{2}\s+\w+)'
)


def parse_gemini_html_stream(zf, html_path):
    """Stream through a large HTML file and extract conversation entries."""
    entry_marker = b'outer-cell mdl-cell mdl-cell--12-col mdl-shadow--2dp'
    end_marker = b'</div></div></div>'
    
    entries = []
    
    with zf.open(html_path) as f:
        buffer = b''
        in_entry = False
        entry_buffer = b''
        entry_count = 0
        
        while True:
            chunk = f.read(65536)
            if not chunk:
                break
            
            buffer += chunk
            
            while True:
                if not in_entry:
                    idx = buffer.find(entry_marker)
                    if idx >= 0:
                        in_entry = True
                        entry_buffer = buffer[idx:]
                        buffer = buffer[idx+100:]
                    else:
                        buffer = buffer[-500:] if len(buffer) > 500 else buffer
                        break
                else:
                    end_idx = buffer.find(end_marker)
                    if end_idx >= 0:
                        entry_buffer += buffer[:end_idx + 18]
                        buffer = buffer[end_idx + 18:]
                        in_entry = False
                        entry_count += 1
                        
                        try:
                            text = entry_buffer.decode('utf-8', errors='replace')
                            entry = parse_single_entry(text)
                            if entry:
                                entries.append(entry)
                        except:
                            pass
                        
                        entry_buffer = b''
                    else:
                        entry_buffer += buffer
                        buffer = b''
                        if len(entry_buffer) > 500000:
                            try:
                                text = entry_buffer.decode('utf-8', errors='replace')
                                entry = parse_single_entry(text)
                                if entry:
                                    entries.append(entry)
                            except:
                                pass
                            entry_buffer = b''
                            in_entry = False
                            entry_count += 1
                        break
    
    return entries


def parse_single_entry(html):
    """Parse a single Gemini conversation entry."""
    # Structure: header + prompt + date + response
    # The prompt starts after the header line
    
    # Find the date - it separates prompt from response
    date_match = DATE_PATTERN.search(html)
    if not date_match:
        return None
    
    date_str = date_match.group(1)
    
    # Everything before the date
    before_date = html[:date_match.start()]
    
    # Remove the header: "outer-cell...Aplicaciones de Gemini<br>" or "Modo IA<br>"
    # Header pattern: everything up to and including the first <br> after the title
    header_match = re.search(r'<br>\s*</p>', before_date)
    if header_match:
        before_date = before_date[header_match.end():]
    
    # Now clean the prompt
    prompt = re.sub(r'<[^>]+>', ' ', before_date)
    prompt = re.sub(r'&\w+;', ' ', prompt)
    prompt = re.sub(r'\s+', ' ', prompt).strip()
    # Remove the "Buscaste" or "Hiciste la petici" prefix that Modo IA has
    prompt = re.sub(r'^(Buscaste|Hiciste la petici[oó]n|You made the query|You asked)\s*', '', prompt)
    prompt = re.sub(r'^(Modo IA|Aplicaciones de Gemini)\s*', '', prompt)
    
    if not prompt or len(prompt) < 5:
        return None
    
    # Everything after the date is the response
    after_date = html[date_match.end():]
    # Skip the <br> tags after the date
    after_date = re.sub(r'^\s*<br[^>]*>', '', after_date)
    
    # Extract response text from HTML
    response = re.sub(r'<[^>]+>', '\n', after_date)
    response = re.sub(r'&\w+;', ' ', response)
    response = re.sub(r'\n{3,}', '\n\n', response)
    response = '\n'.join(line.strip() for line in response.split('\n') if line.strip())
    
    if not response or len(response) < 10:
        return None
    
    return {
        'date': date_str,
        'prompt': prompt,
        'response': response,
        'type': 'gemini_conversation',
    }


def find_gemini_takeouts(exports_dir='exports'):
    """Find all Gemini takeout ZIPs in exports directory."""
    exports = Path(exports_dir)
    if not exports.exists():
        return []
    
    takeouts = []
    for zf in exports.glob('takeout-*.zip'):
        if zf.is_file():
            takeouts.append(zf)
    for zf in exports.glob('*gemini*.zip'):
        if zf.is_file() and zf not in takeouts:
            takeouts.append(zf)
    
    return sorted(takeouts)


def save_conversations(entries, output_path='data/gemini_conversations.json'):
    """Save extracted conversations as JSON."""
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(entries, f, indent=2, ensure_ascii=False)
    print(f'Saved {len(entries)} conversations to {output_path}')


if __name__ == '__main__':
    import sys
    exports_dir = sys.argv[1] if len(sys.argv) > 1 else 'exports'
    
    takeouts = find_gemini_takeouts(exports_dir)
    if not takeouts:
        print(f'No Gemini takeout ZIPs found in {exports_dir}/')
        sys.exit(0)
    
    print(f'Found {len(takeouts)} takeout ZIP(s):')
    for t in takeouts:
        print(f'  {t} ({t.stat().st_size/1e6:.1f} MB)')
    print()
    
    all_entries = []
    for zf in takeouts:
        entries = extract_from_gemini_takeout(zf)
        all_entries.extend(entries)
        print()
    
    print(f'\n{"="*60}')
    print(f'TOTAL: {len(all_entries)} Gemini conversations extracted')
    print(f'{"="*60}')
    
    # Save all conversations
    if all_entries:
        save_conversations(all_entries)
        # Also save a readable sample
        sample = all_entries[:5]
        Path('data/gemini_sample.json').write_text(
            json.dumps(sample, indent=2, ensure_ascii=False), encoding='utf-8'
        )
        print(f'Saved sample to data/gemini_sample.json')
