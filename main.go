package main

import (
	"fmt"

	"github.com/gocolly/colly/v2"
)

func main() {
	// 1. Initialize the Collector
	c := colly.NewCollector(
		colly.AllowedDomains("books.toscrape.com"), // Limit crawling to this domain
		colly.MaxDepth(2),                          // How many links deep to follow
	)

	// 2. On every <a> element with an href attribute
	c.OnHTML("a[href]", func(e *colly.HTMLElement) {
		link := e.Attr("href")
		// Convert relative paths to absolute URLs and visit
		e.Request.Visit(e.Request.AbsoluteURL(link))
	})

	// 3. Extract data (e.g., page titles)
	c.OnHTML("title", func(e *colly.HTMLElement) {
		fmt.Println("Found Page:", e.Text)
	})

	// 4. Log the requests
	c.OnRequest(func(r *colly.Request) {
		fmt.Println("Visiting", r.URL.String())
	})

	// 5. Start the crawler
	c.Visit("https://books.toscrape.com/")
}
