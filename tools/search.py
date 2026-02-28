from ddgs import DDGS

def web_search(query):
    results = []
    with DDGS() as ddgs:
        for r in ddgs.text(query, max_results=5):
            results.append({
                "title": r["title"],
                "body": r["body"],
                "href": r["href"]
            })
    return results