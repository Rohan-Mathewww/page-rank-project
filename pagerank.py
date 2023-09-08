import os
import random
import re
import sys

#PageRank (i.e., the proportion of all the samples that corresponded to that page).

DAMPING = 0.85 # damping factor to find the transition
SAMPLES = 100000 # number of samples taken to find the page rank via sampling
CONVERGENCE = 0.001 # the change in page ranks should be less than this value


def main():
    if len(sys.argv) != 2:
        sys.exit("Usage: python pagerank.py corpus")
    corpus = crawl(sys.argv[1])
    
    for page in sorted(corpus):
        print(f"  {page}: {corpus[page]}")
    
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
    page_transitionprob_dict = dict()
    linked_pages=tuple()
    if page in corpus:
        linked_pages = tuple(corpus[page])
    # if page isnt in the corpus, it raises an error
    else:
        # print("Not a valid page :(")
        sys.exit("Not a valid page :(")
    
    # loop for finding the page rank probability of each page wrt current page
    for i in corpus:
        if i in linked_pages:
            probability = (1-damping_factor)/len(corpus)+(damping_factor/len(linked_pages))
        else:
            probability = (1-damping_factor)/len(corpus)
        
        # page_transitionprob_dict[i]=round(probability,6) #rounding off to 4 decimal points
        page_transitionprob_dict[i]= probability # for now im removing the rounding off part
    
    return page_transitionprob_dict


def sample_pagerank(corpus, damping_factor, n):
    """
    Return PageRank values for each page by sampling `n` pages
    according to transition model, starting with a page at random.

    Return a dictionary where keys are page names, and values are
    their estimated PageRank value (a value between 0 and 1). All
    PageRank values should sum to 1.
    """
    #the following function randomly selects the starting page given the keys of the corpus
    def pick_starting_page(keys):
        keys = list(keys)
        # print(f" the keys are\n{keys}")
        return keys[random.randint(0,len(keys)-1)]

    
    #the following function is to generate a page based on the probability distribution of pages in the transition model
    #remember to make sure that the random should be less than the total cumulative probability
    def simulate_random_event(page_transitionprob_dict):
        cumulative_prob=0.0
        random_value = random.random()
        sampled_page=""
        for page,probability in page_transitionprob_dict.items():
            cumulative_prob+=probability
            if random_value<=cumulative_prob:
                sampled_page=page
                break
        else:
            sampled_page=pick_starting_page(page_transitionprob_dict.keys())
        """
        if it enters the else block above it means that the total cumulative probability is less than random_value
        this might be because any rounding off in probability values while finding the transition_model
        but the probability of this happening is very low
        so for now if this happens well just uniformly randomly select one of the pages in the corpus
        """

        return sampled_page


    sample_count_dict = dict.fromkeys(corpus.keys(),0) # a dictionary containing page:no of samples of the page
    current_page = ""
    
    for i in range(n):
        if i==0:
            #we have to randomly allot the first page and add one to that page in sample_count.
            current_page=pick_starting_page(list(corpus.keys()))
        else:
            #we have to allot the next page based on the transition_model and add one to that page in sample_count.
            current_page = simulate_random_event(transition_model(corpus,current_page,damping_factor))
        
        sample_count_dict[current_page]+=1
    
    for i in sample_count_dict:
        sample_count_dict[i]/= n
    
    return sample_count_dict
    

def iterate_pagerank(corpus, damping_factor):
    """
    Return PageRank values for each page by iteratively updating
    PageRank values until convergence.

    Return a dictionary where keys are page names, and values are
    their estimated PageRank value (a value between 0 and 1). All
    PageRank values should sum to 1.
    """
    
    # function that recieves the old pageranks and calculates the new pageranks
    def find_new_pagerank(corpus, old_dict, damping_factor):
        page_names = list(corpus.keys())
        no_of_pages = len(corpus)
        new_dict = dict()
        
        for p in old_dict:
            # to check for pages that link to page p
            # if the page links to nothing, we interpret it as linking to all pages including itself
            
            #later, we can modify this so that we won't have to keep finding links_to_p everytime, by passing it in
            links_to_p = tuple(i for i in corpus if(p in corpus[i] or not corpus[i]))

            no_of_links_to_p = len(links_to_p)
            
            if no_of_links_to_p > 0:
                new_dict[p] = (1-damping_factor)/no_of_pages

                for j in links_to_p:
                    no_of_pages_from_j = len(corpus[j])
                    if not no_of_pages_from_j:
                        no_of_pages_from_j = len(corpus)
                    
                    new_dict[p] += damping_factor * (old_dict[j]/no_of_pages_from_j)
                '''
                the following line is the ternary version of code above
                new_dict[p] = (1-damping_factor)/no_of_pages + \
                    damping_factor* sum(old_dict[j]/len(corpus[j]) if len(corpus[j]) else len(corpus) for j in links_to_p)
                '''
                
                # the formula above is from Pr(p)=(1-d)/n + d* summation_of_j(Pr(j)/number of links to j)
            
            else:
                new_dict[p] = (1-damping_factor)/no_of_pages
        
        
        return new_dict
    
    is_convergent = True #to implement verification by contradiction
    current_dict = dict.fromkeys(corpus.keys(),1/len(corpus))
    
    while True:
        #print(current_dict)
        new_dict = find_new_pagerank(corpus, current_dict, damping_factor)

        for i in current_dict:
            if abs(current_dict[i]-new_dict[i]) > CONVERGENCE:
                is_convergent = False
                break
        if is_convergent:
            break

        current_dict = new_dict
        is_convergent = True
    
    return current_dict
        
        

if __name__ == "__main__":
    main()
