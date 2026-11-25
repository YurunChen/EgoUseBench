"""Freeze the verified place listings used by visual place-anchor tasks."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
import urllib.request
from urllib.parse import quote_plus
from pathlib import Path
from typing import Any

from PIL import Image

from tools.catalog.freeze_catalog_order import sort_frozen


RETRIEVED_AT = "2026-08-30T03:30:00Z"
USER_AGENT = "egoCUAbench-place-anchor-freezer/1.0"


def _map_search_url(query: str) -> str:
    return f"https://www.google.com/maps/search/?api=1&query={quote_plus(query)}"


def _coordinate_map_url(latitude: float, longitude: float) -> str:
    zoom = 16
    latitude = max(-85.05112878, min(85.05112878, latitude))
    scale = 2**zoom
    x = int((longitude + 180.0) / 360.0 * scale)
    radians = math.radians(latitude)
    y = int((1.0 - math.asinh(math.tan(radians)) / math.pi) / 2.0 * scale)
    return f"https://tile.openstreetmap.org/{zoom}/{x}/{y}.png"


def _mapped_listing(
    place_id: str,
    name: str,
    kind: str,
    summary: str,
    address: str,
    latitude: float,
    longitude: float,
    city: str,
    official_page_url: str,
) -> dict[str, Any]:
    """Build one source-bound listing with a neutral coordinate map image.

    The published address is the decision-bearing fact. Coordinates only center
    the map representation and are never used to distinguish an answer.
    """

    return {
        "place_id": place_id,
        "name": name,
        "kind": kind,
        "summary": summary,
        "address": address,
        "location": {"latitude": latitude, "longitude": longitude},
        "facts": {"listed_city": city},
        "route_options": {"driving": "Directions available"},
        "page_url": _map_search_url(f"{name} {address}"),
        "official_page_url": official_page_url,
        "image_url": _coordinate_map_url(latitude, longitude),
        "source_note": (
            "The linked public locator or current venue page supplies the listing name "
            "and published address. The primary image is a coordinate-backed address-area "
            "map; task decisions use the published listing details, not pin precision."
        ),
    }


PLACES: tuple[dict[str, Any], ...] = (
    _mapped_listing(
        "l_9e69ec9bad3e",
        "Cocoberry — Punjabi Bagh",
        "dessert_shop",
        "Cocoberry frozen-yogurt shop in Punjabi Bagh, New Delhi.",
        "Shop No 2, NWA Club Road, Punjabi Bagh, New Delhi 110026, India",
        28.672995,
        77.1461239,
        "New Delhi",
        "https://www.cocoberryindia.com/",
    ),
    _mapped_listing(
        "l_9de26101a6c4",
        "Cocoberry — Elan Epic Mall",
        "dessert_shop",
        "Cocoberry frozen-yogurt shop in Elan Epic Mall, Gurugram.",
        "Shop No 58A, Lower Ground Floor, Elan Epic Mall, Sector 70, Gurugram, Haryana, India",
        28.4100,
        77.0140,
        "Gurugram",
        "https://www.cocoberryindia.com/",
    ),
    _mapped_listing(
        "l_5a58bbbdcbe7",
        "Cocoberry — West Patel Nagar",
        "dessert_shop",
        "Cocoberry frozen-yogurt shop in West Patel Nagar, New Delhi.",
        "Shop 37, Ground Floor, Y-Block, West Patel Nagar, New Delhi 110008, India",
        28.6519279,
        77.1629230,
        "New Delhi",
        "https://www.cocoberryindia.com/",
    ),
    _mapped_listing(
        "l_8c5b3f4c83b1",
        "Cocoberry — Jamia Nagar",
        "dessert_shop",
        "Cocoberry frozen-yogurt shop near Tikona Park in Jamia Nagar, New Delhi.",
        "Shop No B-11, C-84, Lower Ground Floor, Tikona Park, Jamia Nagar, New Delhi 110025, India",
        28.5621907,
        77.2879148,
        "New Delhi",
        "https://www.cocoberryindia.com/",
    ),
    _mapped_listing(
        "l_179fdee17bfa",
        "Cocoberry — Noida",
        "dessert_shop",
        "Cocoberry frozen-yogurt shop in Sector 29, Noida.",
        "Sector 29, Near HDFC Bank, Noida, Uttar Pradesh, India",
        28.5708315,
        77.3358207,
        "Noida",
        "https://www.cocoberryindia.com/",
    ),
    _mapped_listing(
        "l_535850194b8d",
        "Cocoberry — Ballygunge",
        "dessert_shop",
        "Cocoberry frozen-yogurt shop in Ballygunge, Kolkata.",
        "10, Gurusaday Road, Ballygunge, Kolkata, West Bengal, India",
        22.5347917,
        88.3614931,
        "Kolkata",
        "https://www.cocoberryindia.com/",
    ),
    _mapped_listing(
        "l_abcec1a0a551",
        "Cocoberry — Hemanta Mukhopadhyay Sarani",
        "dessert_shop",
        "Cocoberry frozen-yogurt shop on Hemanta Mukhopadhyay Sarani, Kolkata.",
        "P-244A, Hemanta Mukhopadhyay Sarani, Kolkata, West Bengal, India",
        22.5044355,
        88.3721663,
        "Kolkata",
        "https://www.cocoberryindia.com/",
    ),
    _mapped_listing(
        "l_ee0d6e957c3a",
        "Cocoberry — Dimapur Chekiye",
        "dessert_shop",
        "Cocoberry frozen-yogurt shop in Chekiye, Dimapur.",
        "3rd Mile, Chekiye, Dimapur, Nagaland 797115, India",
        25.8750179,
        93.7555637,
        "Dimapur",
        "https://www.cocoberryindia.com/",
    ),
    _mapped_listing(
        "l_3544384966d9",
        "Cocoberry — Dimapur Central Plaza",
        "dessert_shop",
        "Cocoberry frozen-yogurt shop in Central Plaza, Dimapur.",
        "Central Plaza, Nyamo Lotha Road, Dimapur, Nagaland, India",
        25.9041378,
        93.7177609,
        "Dimapur",
        "https://www.cocoberryindia.com/",
    ),
    _mapped_listing(
        "l_7ce433380ded",
        "Cocoberry — Kohima",
        "dessert_shop",
        "Cocoberry frozen-yogurt shop in central Kohima.",
        "Opposite United Traders, Main Town, Kohima, Nagaland, India",
        25.6618755,
        94.1019156,
        "Kohima",
        "https://www.cocoberryindia.com/",
    ),
    _mapped_listing(
        "l_7ef0d8234167",
        "Tamimi Markets — Al Aarid Town Center Mall",
        "grocery_store",
        "Tamimi Markets supermarket in Al Aarid, Riyadh.",
        "Town Center Mall, Abdullah bin Al-Aas Street, Al Aarid, Riyadh 13338, Saudi Arabia",
        24.8935300,
        46.6060273,
        "Riyadh",
        "https://shop.tamimimarkets.com/en/contact-us",
    ),
    _mapped_listing(
        "l_7c5efd718c43",
        "Tamimi Markets — Al Olaya",
        "grocery_store",
        "Tamimi Markets supermarket in Al Olaya, Riyadh.",
        "Al Olaya, Riyadh, Saudi Arabia",
        24.6981993,
        46.6821165,
        "Riyadh",
        "https://shop.tamimimarkets.com/en/contact-us",
    ),
    _mapped_listing(
        "l_c336af58c9e0",
        "Tamimi Markets — Al Malqa",
        "grocery_store",
        "Tamimi Markets supermarket on Anas Ibn Malik Road in Al Malqa, Riyadh.",
        "Anas Ibn Malik Road, Al Malqa, Riyadh 13522, Saudi Arabia",
        24.7980000,
        46.6090000,
        "Riyadh",
        "https://shop.tamimimarkets.com/en/contact-us",
    ),
    _mapped_listing(
        "l_c21ccba57e32",
        "Tamimi Markets — Irqah Plaza",
        "grocery_store",
        "Tamimi Markets supermarket in Irqah Plaza, Diriyah.",
        "Prince Mishaal Ibn Abdulaziz Road, Irqah Plaza, Diriyah 12583, Saudi Arabia",
        24.6841460,
        46.5826115,
        "Diriyah",
        "https://shop.tamimimarkets.com/en/contact-us",
    ),
    _mapped_listing(
        "l_85facf32b5c2",
        "Tamimi Markets — Al Narjes",
        "grocery_store",
        "Tamimi Markets supermarket in Al Narjes, Riyadh.",
        "Al Narjes, Riyadh, Saudi Arabia",
        24.8257479,
        46.6722626,
        "Riyadh",
        "https://shop.tamimimarkets.com/en/contact-us",
    ),
    _mapped_listing(
        "l_6330a1b1bb23",
        "Tamimi Markets — Al Nuzha Plaza",
        "grocery_store",
        "Tamimi Markets supermarket in Al Nuzha, Riyadh.",
        "Prince Muqrin Bin Abdulaziz Road, Al Nuzha, Riyadh 12474, Saudi Arabia",
        24.7569440,
        46.7075000,
        "Riyadh",
        "https://shop.tamimimarkets.com/en/contact-us",
    ),
    _mapped_listing(
        "l_a569e3dd0eb5",
        "Tamimi Markets — Al Yasmin",
        "grocery_store",
        "Tamimi Markets supermarket in Al Yasmin, Riyadh.",
        "King Abdulaziz Branch Road, Al Yasmin, Riyadh 13322, Saudi Arabia",
        24.8203042,
        46.6425171,
        "Riyadh",
        "https://shop.tamimimarkets.com/en/contact-us",
    ),
    _mapped_listing(
        "l_fde602fdd554",
        "Tamimi Markets — As Sulimaniyah",
        "grocery_store",
        "Tamimi Markets supermarket in As Sulimaniyah, Riyadh.",
        "As Sulimaniyah, Riyadh 12234, Saudi Arabia",
        24.7128984,
        46.7002912,
        "Riyadh",
        "https://shop.tamimimarkets.com/en/contact-us",
    ),
    _mapped_listing(
        "l_624fd0d03252",
        "Tamimi Markets — Al Hamra",
        "grocery_store",
        "Tamimi Markets supermarket in Al Hamra, Riyadh.",
        "Al Hamra, Riyadh, Saudi Arabia",
        24.7761692,
        46.7526513,
        "Riyadh",
        "https://shop.tamimimarkets.com/en/contact-us",
    ),
    _mapped_listing(
        "l_6fd82e245b2c",
        "Tamimi Markets — Ar Rawda",
        "grocery_store",
        "Tamimi Markets supermarket in Ar Rawda, Riyadh.",
        "Ar Rawda, Riyadh 13211, Saudi Arabia",
        24.7340640,
        46.7671930,
        "Riyadh",
        "https://shop.tamimimarkets.com/en/contact-us",
    ),
    _mapped_listing(
        "l_74533f635a43",
        "Tamimi Markets — Corniche Road",
        "grocery_store",
        "Tamimi Markets supermarket on Corniche Road in Al Khobar.",
        "Corniche Road, Al Khobar 34413, Saudi Arabia",
        26.3057088,
        50.2273363,
        "Al Khobar",
        "https://shop.tamimimarkets.com/en/contact-us",
    ),
    _mapped_listing(
        "l_ab4dd044df55",
        "Country Moon Winery",
        "winery",
        "Estate winery and tasting room in Noblesville, Indiana.",
        "16222 Prairie Baptist Road, Noblesville, IN 46060",
        40.0257656,
        -85.9004073,
        "Noblesville",
        "https://countrymoonwinery.com/",
    ),
    _mapped_listing(
        "l_706b5a2f730e",
        "Woolly Yak Ranch & Winery",
        "winery",
        "Working ranch, winery, and tasting venue in Arcadia, Indiana.",
        "3124 E 266th Street, Arcadia, IN 46030",
        40.1757403,
        -86.0380660,
        "Arcadia",
        "https://www.woollyyak.com/plan-your-visit",
    ),
    _mapped_listing(
        "l_5b1d3d657284",
        "Harmony Winery",
        "winery",
        "Winery and tasting room in Knightstown, Indiana.",
        "18 N Jefferson Street, Knightstown, IN 46148",
        39.7993428,
        -85.5266767,
        "Knightstown",
        "https://www.harmonywinery.com/",
    ),
    _mapped_listing(
        "l_ab4b37daee91",
        "Easley Winery",
        "winery",
        "Winery and tasting room in downtown Indianapolis.",
        "205 N College Avenue, Indianapolis, IN 46202",
        39.7698290,
        -86.1451695,
        "Indianapolis",
        "https://www.easleywinery.com/contact/",
    ),
    _mapped_listing(
        "l_e880fa99d1d4",
        "Buck Creek Winery",
        "winery",
        "Winery and commercial vineyard in Indianapolis.",
        "11747 Indian Creek Road South, Indianapolis, IN 46259",
        39.6747433,
        -85.9581840,
        "Indianapolis",
        "https://buckcreekwinery.com/vineyard",
    ),
    _mapped_listing(
        "l_055abce72b74",
        "Brandywine Creek Winery & Vineyards",
        "winery",
        "Vineyard, winery, and tasting room southeast of Indianapolis.",
        "12524 Southeastern Avenue, Indianapolis, IN 46259",
        39.6646440,
        -85.9430980,
        "Indianapolis",
        "https://www.brandywinecreekvineyards.com/",
    ),
    _mapped_listing(
        "l_ba0b4478a28f",
        "Lanthier Winery & Distillery",
        "winery",
        "Winery, distillery, gardens, and tasting venue in Madison, Indiana.",
        "123 Mill Street, Madison, IN 47250",
        38.7365000,
        -85.3800000,
        "Madison",
        "https://www.lanthierwinery.com/",
    ),
    _mapped_listing(
        "l_a7b477b54f16",
        "Oliver Winery",
        "winery",
        "Winery and tasting room in Bloomington, Indiana.",
        "200 East Winery Road, Bloomington, IN 47404",
        39.2824442,
        -86.5198669,
        "Bloomington",
        "https://www.oliverwinery.com/visit",
    ),
    _mapped_listing(
        "l_5e4b8371e066",
        "Cedar Creek Winery",
        "winery",
        "Licensed farm winery and tasting venue in Martinsville, Indiana.",
        "3820 Leonard Road, Martinsville, IN 46151",
        39.4650000,
        -86.3520000,
        "Martinsville",
        "https://www.in.gov/atc/files/2026.3-LB.pdf",
    ),
    _mapped_listing(
        "l_1ef58ab889d9",
        "Mallow Run Winery",
        "winery",
        "Farm winery, tasting room, and event venue in Bargersville, Indiana.",
        "6964 West Whiteland Road, Bargersville, IN 46106",
        39.5469473,
        -86.2336262,
        "Bargersville",
        "https://www.mallowrun.com/location-directions/",
    ),
    {
        "place_id": "l_0b409629a567",
        "name": "Cocoberry — GK1",
        "kind": "dessert_shop",
        "summary": "Cocoberry frozen-yogurt shop in Greater Kailash 1, New Delhi.",
        "address": "S-13, M Block Market, Greater Kailash 1, New Delhi, India",
        "location": {"latitude": 28.5502, "longitude": 77.23315},
        "facts": {"listed_city": "New Delhi"},
        "route_options": {"driving": "Directions available"},
        "page_url": _map_search_url("Cocoberry GK1 New Delhi"),
        "official_page_url": "https://www.cocoberryindia.com/",
        "image_url": _coordinate_map_url(28.5502, 77.23315),
        "source_note": "The official Cocoberry store list supplies the GK1 branch name and address; the primary image is a coordinate-backed map for that address.",
    },
    {
        "place_id": "l_2f1218f0fcad",
        "name": "Cocoberry — GK2",
        "kind": "dessert_shop",
        "summary": "Cocoberry frozen-yogurt shop in Greater Kailash 2, New Delhi.",
        "address": "E-586, Greater Kailash 2, New Delhi 110048, India",
        "location": {"latitude": 28.5369298, "longitude": 77.2394367},
        "facts": {"listed_city": "New Delhi"},
        "route_options": {"driving": "Directions available"},
        "page_url": _map_search_url("Cocoberry GK2 New Delhi"),
        "official_page_url": "https://www.cocoberryindia.com/",
        "image_url": _coordinate_map_url(28.5369298, 77.2394367),
        "source_note": "The official Cocoberry store list supplies the GK2 branch name and address; the primary image is a coordinate-backed map for that address.",
    },
    {
        "place_id": "l_19c411b37471",
        "name": "Cocoberry — Rajouri Garden",
        "kind": "dessert_shop",
        "summary": "Cocoberry frozen-yogurt shop in Rajouri Garden, New Delhi.",
        "address": "Shop J13/44A, Ground Floor, Rajouri Garden, New Delhi 110027, India",
        "location": {"latitude": 28.6451121, "longitude": 77.1239332},
        "facts": {"listed_city": "New Delhi"},
        "route_options": {"driving": "Directions available"},
        "page_url": _map_search_url("Cocoberry Rajouri Garden New Delhi"),
        "official_page_url": "https://www.cocoberryindia.com/",
        "image_url": _coordinate_map_url(28.6451121, 77.1239332),
        "source_note": "The official Cocoberry store list supplies the Rajouri Garden branch name and address; the primary image is a coordinate-backed map for that address.",
    },
    {
        "place_id": "l_793df5699dfc",
        "name": "Giani Ice Cream — Defence Colony",
        "kind": "dessert_shop",
        "summary": "Giani ice-cream shop in Defence Colony Main Market, New Delhi.",
        "address": "MCD Booth, Main Market, Defence Colony, New Delhi 110024, India",
        "location": {"latitude": 28.5742, "longitude": 77.2309},
        "facts": {"listed_city": "New Delhi"},
        "route_options": {"driving": "Directions available"},
        "page_url": _map_search_url("Giani Ice Cream Defence Colony New Delhi"),
        "official_page_url": "https://gianiicecream.in/wp-content/uploads/2024/10/GMB_Data.pdf",
        "image_url": _coordinate_map_url(28.5742, 77.2309),
        "source_note": "Giani's published branch list supplies the Defence Colony shop name and address; the primary image is a coordinate-backed map for that address.",
    },
    {
        "place_id": "l_1aedc3861c9a",
        "name": "Tamimi Markets — An Nakheel",
        "kind": "grocery_store",
        "summary": "Tamimi Markets supermarket in the An Nakheel area of Riyadh.",
        "address": "An Nakheel, Riyadh 12572, Saudi Arabia",
        "location": {"latitude": 24.7414612, "longitude": 46.6033446},
        "facts": {"listed_city": "Riyadh"},
        "route_options": {"driving": "Directions available"},
        "page_url": _map_search_url("Tamimi Markets An Nakheel Riyadh"),
        "official_page_url": "https://www.tamimimarkets.com/StoreLocator",
        "image_url": _coordinate_map_url(24.7414612, 46.6033446),
        "source_note": "The official Tamimi store locator lists An Nakheel; OpenStreetMap supplies the place coordinate used by the location-bearing map image.",
    },
    {
        "place_id": "l_0b7135e02647",
        "name": "Daniel's Vineyard",
        "kind": "winery",
        "summary": "Full-production winery and event venue in McCordsville, Indiana.",
        "address": "9061 N 700 W, McCordsville, IN 46055",
        "location": {"latitude": 39.9162755, "longitude": -85.9335691},
        "facts": {"listed_city": "McCordsville"},
        "route_options": {"driving": "Directions available"},
        "page_url": "https://www.danielsvineyard.com/",
        "image_url": _coordinate_map_url(39.9162755, -85.9335691),
        "source_note": "The official winery page supplies the name and address; OpenStreetMap supplies the place coordinate used by the location-bearing map image.",
    },
    {
        "place_id": "l_8bb4f46941de",
        "name": "Spencer Farm Winery",
        "kind": "winery",
        "summary": "Farm winery and tasting room in Noblesville, Indiana.",
        "address": "7015 E 161st Street, Noblesville, IN 46062",
        "location": {"latitude": 40.0218942, "longitude": -86.047136},
        "facts": {"listed_city": "Noblesville"},
        "route_options": {"driving": "Directions available"},
        "page_url": "https://spencerfarmwinery.com/contact-us",
        "image_url": _coordinate_map_url(40.0218942, -86.047136),
        "source_note": "The official winery page supplies the name and address; OpenStreetMap supplies the place coordinate used by the location-bearing map image.",
    },
    {
        "place_id": "l_e15b6d6b1de7",
        "name": "Urban Vines Winery & Brewery",
        "kind": "winery",
        "summary": "Winery, brewery, tasting room, and vineyard in Westfield, Indiana.",
        "address": "303 E 161st Street, Westfield, IN 46074",
        "location": {"latitude": 40.0206524, "longitude": -86.1535166},
        "facts": {"listed_city": "Westfield"},
        "route_options": {"driving": "Directions available"},
        "page_url": "https://urban-vines.com/pages/visit-us",
        "image_url": _coordinate_map_url(40.0206524, -86.1535166),
        "source_note": "The official winery page supplies the name and address; OpenStreetMap supplies the place coordinate used by the location-bearing map image.",
    },
    {
        "place_id": "l_6e86c83dabd9",
        "name": "Peace Water Winery — Carmel",
        "kind": "winery",
        "summary": "Peace Water Winery tasting room in the Carmel Arts & Design District.",
        "address": "37 West Main Street, Carmel, IN 46032",
        "location": {"latitude": 39.97819, "longitude": -86.128},
        "facts": {"listed_city": "Carmel"},
        "route_options": {"walking": "15 min", "driving": "8 min"},
        "page_url": "https://www.peacewaterwinery.com/locations#carmel",
        "image_url": "https://www.peacewaterwinery.com/uploads/b/df2ddf60bc33a125750f6b74f7b74d83b7f8413c386b1514648ae89deb2e3292/IMG_20200619_073401_1685145998.jpg",
        "source_note": "Official locations page supplies the branch name, address, coordinates, and location-gallery image.",
    },
    {
        "place_id": "l_72ddaf59ea9d",
        "name": "Peace Water Winery — Fishers",
        "kind": "winery",
        "summary": "Peace Water Winery tasting room at The Yard in Fishers District.",
        "address": "11547 Yard Street, Fishers, IN 46038",
        "location": {"latitude": 39.95573, "longitude": -86.00066},
        "facts": {"listed_city": "Fishers"},
        "route_options": {"walking": "21 min", "driving": "6 min"},
        "page_url": "https://www.peacewaterwinery.com/locations#fishers",
        "image_url": "https://www.peacewaterwinery.com/uploads/b/df2ddf60bc33a125750f6b74f7b74d83b7f8413c386b1514648ae89deb2e3292/IMG_20200619_073413_1685140352.jpg",
        "source_note": "Official locations page supplies the branch name, address, coordinates, and location-gallery image.",
    },
    {
        "place_id": "l_cb2e2cee82c4",
        "name": "Tamimi Markets Express — Discovery Square",
        "kind": "grocery_store",
        "summary": "Tamimi minimarket at Discovery Square, also shown under its earlier Express name.",
        "address": "Discovery Square TC-7445, KAUST, Thuwal 23955, Saudi Arabia",
        "location": {"latitude": 22.311619, "longitude": 39.1013193},
        "facts": {"listed_city": "Thuwal"},
        "route_options": {"walking": "8 min", "driving": "9 min"},
        "page_url": "https://cc.kaust.edu.sa/services/food-retail/retail-outlets#tamimi-minimarket",
        "image_url": "https://cc.kaust.edu.sa/images/default-source/retail-shops-2024/retail-web-2024-(4).jpg?sfvrsn=90e4701c_0",
        "source_note": "Official KAUST retail page supplies the current minimarket name, Discovery Square unit, and image; the official 2022 notice records the visible Express alias.",
    },
    {
        "place_id": "l_ef8650dd668b",
        "name": "Tamimi Supermarket — KAUST Health",
        "kind": "grocery_store",
        "summary": "Full-size Tamimi supermarket near KAUST Health and the main bus shelter.",
        "address": "Between KAUST Health and the main bus shelter, KAUST, Thuwal 23955, Saudi Arabia",
        "location": {"latitude": 22.314816, "longitude": 39.104646},
        "facts": {"listed_city": "Thuwal"},
        "route_options": {"walking": "4 min", "driving": "10 min"},
        "page_url": "https://cc.kaust.edu.sa/services/food-retail/retail-outlets#tamimi-supermarket",
        "image_url": "https://cc.kaust.edu.sa/images/default-source/2024-clubs/retail-web-2024-1.png?sfvrsn=fca1d901_0",
        "source_note": "Official KAUST retail page supplies the supermarket name, relative location, and image.",
    },
    {
        "place_id": "l_f360fb464f5e",
        "name": "Tamimi Markets Express — Qurtubah",
        "kind": "grocery_store",
        "summary": "Tamimi Markets Express outlet in Qurtubah, Riyadh.",
        "address": "Prince Saud Bin Mohammed Bin Muqrin Road, Qurtubah, Exit 8, Riyadh 13244, Saudi Arabia",
        "location": {"latitude": 24.8115625, "longitude": 46.7268125},
        "facts": {"listed_city": "Riyadh"},
        "route_options": {"walking": "10 min", "driving": "2 min"},
        "page_url": "https://magicpin.com/saudi-arabia/Riyadh/Qurtubah-Riyadh/Other/Tamimi-Markets-Express/store/2279732",
        "image_url": "https://img.magicpin.com/10617128_store_images_0.webp",
        "source_note": "Verified public branch listing supplies the Express format, Qurtubah address, coordinates, and listing image.",
    },
    {
        "place_id": "l_6f56ee36a2c7",
        "name": "Discovery Square",
        "kind": "shopping_plaza",
        "summary": "Central KAUST community square with shops, food outlets, seating, and event space.",
        "address": "King Abdullah University of Science and Technology, Thuwal 23955, Saudi Arabia",
        "location": {"latitude": 22.3113565, "longitude": 39.1015267},
        "facts": {"listed_city": "Thuwal"},
        "route_options": {"walking": "16 min", "driving": "13 min"},
        "page_url": "https://cc.kaust.edu.sa/spaces/outdoor-spaces/discovery-square",
        "image_url": "https://cc.kaust.edu.sa/images/default-source/spaces/outdoor-spaces/discovery-sq-540x340.jpg?sfvrsn=56ed1d78_0",
        "source_note": "Official KAUST place page supplies the square identity, description, and image.",
    },
    {
        "place_id": "l_93297fefa80d",
        "name": "Cocoberry — Defence Colony",
        "kind": "dessert_shop",
        "summary": "Cocoberry frozen-yogurt shop in Defence Colony, New Delhi.",
        "address": "Shop 1, A-272, Defence Colony, New Delhi 110024, India",
        "location": {"latitude": 28.5746396, "longitude": 77.2292613},
        "facts": {"listed_city": "New Delhi"},
        "route_options": {"walking": "5 min", "driving": "2 min"},
        "page_url": "https://www.zomato.com/ncr/cocoberry-2-defence-colony-new-delhi/photos",
        "image_url": "https://b.zmtcdn.com/data/pictures/chains/7/2067/0b142f7e23ffc2b56103ddf9dc5d8616.jpg",
        "source_note": "Public Defence Colony branch listing supplies the address and branch identity; a Cocoberry gallery image is frozen for the local listing.",
    },
    {
        "place_id": "l_20fb69c46737",
        "name": "Cocoberry — DLF Phase 4",
        "kind": "dessert_shop",
        "summary": "Cocoberry frozen-yogurt shop in Super Mart 1, DLF Phase 4.",
        "address": "A-130, Lower Ground Floor, Super Mart 1, DLF Phase 4, Gurugram 122009, India",
        "location": {"latitude": 28.4623717, "longitude": 77.086965},
        "facts": {"listed_city": "Gurugram"},
        "route_options": {"walking": "8 min", "driving": "3 min"},
        "page_url": "https://www.district.in/dining/ncr/cocoberry-dlf-phase-4-gurgaon",
        "image_url": "https://b.zmtcdn.com/data/pictures/chains/7/2067/a7429422860a71a29a986d94710659ab.jpg",
        "source_note": "Public branch listing supplies the address, branch identity, coordinates, and listing image.",
    },
)


def fetch_jpeg(url: str) -> bytes:
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(request, timeout=30) as response:
        raw = response.read()
    with Image.open(io.BytesIO(raw)) as image:
        output = io.BytesIO()
        image.convert("RGB").save(output, format="JPEG", quality=90)
    return output.getvalue()


def freeze(root: Path) -> dict[str, Any]:
    catalog_path = root / "sites/egomaps/places.json"
    payload = json.loads(catalog_path.read_text())
    by_id = {place["place_id"]: place for place in payload["places"]}

    for source in PLACES:
        place = dict(source)
        image_url = place.pop("image_url")
        page_url = place.pop("page_url")
        official_page_url = place.pop("official_page_url", None)
        source_note = place.pop("source_note")
        photo_path = f"sites/egomaps/static/photos/{place['place_id']}.jpg"
        absolute_photo = root / photo_path
        previous = by_id.get(place["place_id"], {})
        previous_image_url = str((previous.get("source") or {}).get("image_url") or "")
        jpeg = (
            absolute_photo.read_bytes()
            if absolute_photo.exists() and previous_image_url == image_url
            else fetch_jpeg(image_url)
        )
        absolute_photo.parent.mkdir(parents=True, exist_ok=True)
        absolute_photo.write_bytes(jpeg)
        place.update(
            {
                "photo_path": photo_path,
                "source": {
                    "page_url": page_url,
                    **(
                        {"official_page_url": official_page_url}
                        if official_page_url
                        else {}
                    ),
                    "image_url": image_url,
                    "retrieved_at": RETRIEVED_AT,
                    "license_or_access_note": "Upstream listing image frozen for local research evaluation; redistribution rights are not asserted unless the upstream page says otherwise.",
                    "sha256": hashlib.sha256(jpeg).hexdigest(),
                    "access_note": source_note,
                },
            }
        )
        by_id[place["place_id"]] = place

    payload["places"] = sort_frozen(list(by_id.values()), "place_id")
    payload["content_origin"] = (
        "Frozen upstream place listings and images; task-scored state is limited to stable identity and branch fields."
    )
    catalog_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
    return {"updated": len(PLACES), "total": len(payload["places"])}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=".")
    args = parser.parse_args()
    print(json.dumps(freeze(Path(args.root).resolve()), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
