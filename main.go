package main

import (
	"context"
	"fmt"
	"log"
	"time"

	"github.com/chromedp/chromedp"
)

type Books struct {
	AssetID       int
	AssetTitle    string
	Symbol        string
	Category      string
	Price         float64
	PricingSource string
	Currency      string
}

func main() {
	ctx := context.Background()
	// db := database.NewPostgresDB("postgres", "postgres", "127.0.0.1", "asset_db", "6001", "asset")

	// assets, err := db.Query(ctx, "SELECT asset_id, asset_title, symbol, category, price, pricing_source, currency FROM asset.books")
	// if err != nil {
	// 	panic(err)
	// }
	// defer assets.Close()

	// var assetList []Books
	// for assets.Next() {
	// 	var book Books
	// 	err := assets.Scan(&book.AssetID, &book.AssetTitle, &book.Symbol, &book.Category, &book.Price, &book.PricingSource, &book.Currency)
	// 	if err != nil {
	// 		panic(err)
	// 	}
	// 	assetList = append(assetList, book)
	// }

	// for _, asset := range assetList {
	// 	fmt.Printf("Asset ID: %d, Title: %s, Symbol: %s, Category: %s, Price: %.2f, Source: %s, Currency: %s\n",
	// 		asset.AssetID, asset.AssetTitle, asset.Symbol, asset.Category, asset.Price, asset.PricingSource, asset.Currency)
	// }

	// 3. Crawl price for each asset
	// 1. Setup Chrome options (User-Agent is vital to avoid instant blocks)
	opts := append(chromedp.DefaultExecAllocatorOptions[:],
		chromedp.UserAgent("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"),
	)

	allocCtx, cancel := chromedp.NewExecAllocator(context.Background(), opts...)
	defer cancel()

	ctx, cancel = chromedp.NewContext(allocCtx)
	defer cancel()

	// 2. Set a timeout
	ctx, cancel = context.WithTimeout(ctx, 30*time.Second)
	defer cancel()

	var rowsData []map[string]string

	var tmp string

	// 3. Run the tasks
	err := chromedp.Run(ctx,
		chromedp.Navigate(`https://th.tradingview.com/markets/cryptocurrencies/prices-all/`),
		// Wait for the specific table body to load
		chromedp.WaitVisible(`tbody[data-testid="selectable-rows-table-body"]`, chromedp.ByQuery),
		// Use JS to extract the data because it's more reliable for dynamic lists
		chromedp.Evaluate(`
			Array.from(document.querySelectorAll('tr[class="row-RdUXZpkv listRow"]')).map(tr => {
				return {
					symbol: tr.querySelector('.tickerName-GrtoTeat')?.innerText || "",
					price:  tr.querySelector('td:nth-child(3)')?.innerText || ""
				};
			})
		`, &rowsData),

		chromedp.Text(`h2`, &tmp, chromedp.ByQuery),
	)

	fmt.Println("Category Title:", tmp)
	if err != nil {
		log.Fatal("Error crawling:", err)
	}

	// 4. Display results
	for _, data := range rowsData {
		fmt.Printf("Symbol: %-10s | Price: %s\n", data["symbol"], data["price"])
	}

	// 4. Update price in db

	// 5. Repeat every n minutes
}
