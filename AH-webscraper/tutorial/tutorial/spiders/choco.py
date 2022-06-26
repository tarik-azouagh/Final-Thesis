import scrapy


class ChocoSpider(scrapy.Spider):
    name = 'choco'
    allowed_domains = ['ah.nl']
    start_urls = ['https://www.ah.nl/producten/snoep-koek-chips-en-chocolade/chocolade/repen-en-tabletten?page=70']

    #Go into the details pages (find the urls to the detailspages)
    def parse(self, response):
        urls = response.css('div.header_root__22e61 > a::attr(href)').extract()
        for url in urls:
            url = response.urljoin(url)
            yield scrapy.Request(url=url, callback=self.parse_details)

    #Scrape data from detailspages
    def parse_details(self, response):
        yield {
            'product_id': response.css('#start-of-content > div:nth-child(2) > div > div:nth-child(2) > div > p::text').extract(),
            'product_naam': response.css('div.product-card-header_root__1GTl1 > h1 > span::text').extract(),
            'prijs': response.css('div.product-card-hero-price_now__PlF9u > span::text').getall(),
            'kilo_prijs': response.css('div.product-card-header_unitInfo__2ncbP > span::text').extract(),
            'omschrijving': response.css('div.column.xlarge-6.large-8.small-12.xlarge-offset-1 > div:nth-child(1) > div:nth-child(2) > ul > li::text').extract(),
            'inhoud_gewicht': response.css('div.product-info-content-block.product-info-content-block--compact > p::text').extract(),
            'ingredienten': response.css('div.column.xlarge-6.large-8.small-12.xlarge-offset-1 > div:nth-child(2) > p > span').extract(),
            'kenmerken': response.css('div.column.xlarge-6.large-8.small-12.xlarge-offset-1 > div:nth-child(1) > ul > li > p::text').extract()
        }

    #Follow links pagination (does not work because of html AH.nl)
        #next_page_url = response.css('ul.search-pagination_root__11Ik7 > li > a::attr(href)').extract_first()
        #if next_page_url is not None:
        #   yield response.follow(next_page_url, callback=self.parse)
            #next_page_url = response.urljoin(next_page_url)
            #yield scrapy.Request(url=next_page_url, callback=self.parse)