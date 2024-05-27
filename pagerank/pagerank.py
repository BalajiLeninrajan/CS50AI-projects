import os
import random
import re
import sys

DAMPING = 0.85
SAMPLES = 10000


def main():
    if len(sys.argv) != 2:
        sys.exit("Usage: python pagerank.py corpus")
    corpus = crawl(sys.argv[1])
    ranks = sample_pagerank(corpus, DAMPING, SAMPLES)
    print(f"PageRank Results from Sampling (n = {SAMPLES})")
    for page in sorted(ranks):
        print(f"  {page}: {ranks[page]:.4f}")
    ranks = iterate_pagerank(corpus, DAMPING)
    print(f"PageRank Results from Iteration")
    for page in sorted(ranks):
        print(f"  {page}: {ranks[page]:.4f}")


def crawl(directory):
    """
    Parse a directory of HTML pages and check for links to other pages.
    Return a dictionary where each key is a page, and values are
    a list of all other pages in the corpus that are linked to by the page.
    """
    pages = dict()

    # Extract all links from HTML files
    for filename in os.listdir(directory):
        if not filename.endswith(".html"):
            continue
        with open(os.path.join(directory, filename)) as f:
            contents = f.read()
            links = re.findall(r"<a\s+(?:[^>]*?)href=\"([^\"]*)\"", contents)
            pages[filename] = set(links) - {filename}

    # Only include links to other pages in the corpus
    for filename in pages:
        pages[filename] = set(
            link for link in pages[filename]
            if link in pages
        )

    return pages


def transition_model(corpus, page, damping_factor):
    """
    Return a probability distribution over which page to visit next,
    given a current page.

    With probability `damping_factor`, choose a link at random
    linked to by `page`. With probability `1 - damping_factor`, choose
    a link at random chosen from all pages in the corpus.
    """
    res = {}

    if not corpus[page]:
        prob = 1 / len(corpus)
        for key in corpus:
            res[key] = prob
        return res

    prob_from_page = damping_factor / len(corpus[page])
    prob_random = (1 - damping_factor) / len(corpus)
    for key in corpus:
        res[key] = prob_random
        if key in corpus[page]:
            res[key] += prob_from_page

    return res


def sample_pagerank(corpus, damping_factor, n):
    """
    Return PageRank values for each page by sampling `n` pages
    according to transition model, starting with a page at random.

    Return a dictionary where keys are page names, and values are
    their estimated PageRank value (a value between 0 and 1). All
    PageRank values should sum to 1.
    """
    counts = {key: 0 for key in corpus}

    current_page = random.choice(list(corpus))
    counts[current_page] = 1

    for _ in range(n):
        model = transition_model(corpus, current_page, damping_factor)
        keys = []
        values = []
        for key, value in model.items():
            keys.append(key)
            values.append(value)
        current_page = random.choices(keys, weights=values)[0]
        counts[current_page] += 1

    res = {key: value/n for key, value in counts.items()}

    return res


def iterate_pagerank(corpus, damping_factor):
    """
    Return PageRank values for each page by iteratively updating
    PageRank values until convergence.

    Return a dictionary where keys are page names, and values are
    their estimated PageRank value (a value between 0 and 1). All
    PageRank values should sum to 1.
    """
    num_pages = len(corpus)
    max_delta = 1 / num_pages
    current_ranks = {key: 1/num_pages for key in corpus}
    new_ranks = {key: 0 for key in corpus}

    while max_delta > 0.001:
        max_delta = 0
        for page, rank in current_ranks.items():
            new_rank = 0
            for i in corpus:
                if not corpus[i]:
                    new_rank += current_ranks[i] / num_pages
                elif page in corpus[i]:
                    new_rank += current_ranks[i] / len(corpus[i])

            new_rank *= damping_factor
            new_rank += (1 - damping_factor) / num_pages

            new_ranks[page] = new_rank
            nf = sum(new_ranks.values())
            new_ranks = {page: (rank / nf)
                         for page, rank in new_ranks.items()}
            max_delta = max(max_delta, abs(new_rank - rank))

            current_ranks = new_ranks

    return current_ranks


if __name__ == "__main__":
    main()
