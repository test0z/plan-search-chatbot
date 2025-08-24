"""
CORP Plugin for semantic kernel that provides corporation code lookup functionality.
This plugin manages DART corporation codes from CORPCODE.xml in SQLite database.
"""

import os
import sqlite3
import xml.etree.ElementTree as ET
import logging
from typing import List, Dict, Optional, Annotated
from semantic_kernel.functions import kernel_function
import asyncio
from concurrent.futures import ThreadPoolExecutor

logger = logging.getLogger(__name__)


class CORPPlugin:
    """
    Corporation plugin for semantic kernel that provides corp_code lookup functionality.
    Manages DART corporation data from CORPCODE.xml file using SQLite database.
    """

    def __init__(self, db_path: str = "corp_codes.db", xml_path: str = None):
        """
        Initialize the CORPPlugin with SQLite database.

        Args:
            db_path: Path to SQLite database file
            xml_path: Path to CORPCODE.xml file (optional, for initial data loading)
        """
        self.db_path = db_path
        self.xml_path = xml_path or os.path.join(os.getcwd(), "CORPCODE.xml")
        self.executor = ThreadPoolExecutor(max_workers=2)

        # Initialize database
        self._init_database()

        # Load data from XML if database is empty and XML file exists
        if self._is_database_empty() and os.path.exists(self.xml_path):
            logger.info(f"Database is empty. Loading data from {self.xml_path}")
            self._load_xml_to_database()

        logger.info("CORPPlugin initialized with:")
        logger.info(f"  - Database path: {self.db_path}")
        logger.info(f"  - XML path: {self.xml_path}")
        logger.info(f"  - Total companies in DB: {self._get_total_companies()}")

    def _init_database(self):
        """Initialize SQLite database with corporation table."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()

                # Create corporations table
                cursor.execute(
                    """
                    CREATE TABLE IF NOT EXISTS corporations (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        corp_code TEXT NOT NULL UNIQUE,
                        corp_name TEXT NOT NULL,
                        stock_code TEXT,
                        modify_date TEXT,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                """
                )

                # Create indexes for faster search
                cursor.execute(
                    """
                    CREATE INDEX IF NOT EXISTS idx_corp_name 
                    ON corporations(corp_name)
                """
                )

                cursor.execute(
                    """
                    CREATE INDEX IF NOT EXISTS idx_corp_code 
                    ON corporations(corp_code)
                """
                )

                cursor.execute(
                    """
                    CREATE INDEX IF NOT EXISTS idx_stock_code 
                    ON corporations(stock_code)
                """
                )

                conn.commit()
                logger.info("Database initialized successfully")

        except Exception as e:
            logger.error(f"Failed to initialize database: {e}")
            raise

    def _is_database_empty(self) -> bool:
        """Check if the database is empty."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT COUNT(*) FROM corporations")
                count = cursor.fetchone()[0]
                return count == 0
        except Exception as e:
            logger.error(f"Failed to check database status: {e}")
            return True

    def _get_total_companies(self) -> int:
        """Get total number of companies in database."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT COUNT(*) FROM corporations")
                return cursor.fetchone()[0]
        except Exception as e:
            logger.error(f"Failed to get total companies: {e}")
            return 0

    def _load_xml_to_database(self):
        """Load corporation data from CORPCODE.xml to SQLite database."""
        try:
            logger.info(f"Parsing XML file: {self.xml_path}")
            companies = self._parse_corp_code_xml(self.xml_path)

            logger.info(
                f"Found {len(companies)} companies in XML. Inserting to database..."
            )

            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()

                # Insert companies in batches for better performance
                batch_size = 1000
                for i in range(0, len(companies), batch_size):
                    batch = companies[i : i + batch_size]
                    cursor.executemany(
                        """
                        INSERT OR REPLACE INTO corporations 
                        (corp_code, corp_name, stock_code, modify_date)
                        VALUES (?, ?, ?, ?)
                    """,
                        [
                            (
                                comp["corp_code"],
                                comp["corp_name"],
                                comp["stock_code"],
                                comp["modify_date"],
                            )
                            for comp in batch
                        ],
                    )

                    if i % (batch_size * 10) == 0:  # Log every 10 batches
                        logger.info(
                            f"Inserted {min(i + batch_size, len(companies))}/{len(companies)} companies"
                        )

                conn.commit()
                logger.info(
                    f"Successfully loaded {len(companies)} companies to database"
                )

        except Exception as e:
            logger.error(f"Failed to load XML to database: {e}")
            raise

    def _parse_corp_code_xml(self, file_path: str) -> List[Dict[str, str]]:
        """
        Parse CORPCODE.xml file and return list of company information.

        Args:
            file_path: Path to CORPCODE.xml file

        Returns:
            List of company dictionaries with corp_code, corp_name, stock_code, modify_date
        """
        try:
            tree = ET.parse(file_path)
            root = tree.getroot()

            company_list = []
            # Support both <result> and <r> root elements
            list_elements = root.findall("list")

            for company in list_elements:
                corp_code_elem = company.find("corp_code")
                corp_name_elem = company.find("corp_name")
                stock_code_elem = company.find("stock_code")
                modify_date_elem = company.find("modify_date")

                # Skip if essential elements are missing
                if corp_code_elem is None or corp_name_elem is None:
                    continue

                corp_code = corp_code_elem.text.strip() if corp_code_elem.text else ""
                corp_name = corp_name_elem.text.strip() if corp_name_elem.text else ""
                stock_code = (
                    stock_code_elem.text.strip()
                    if stock_code_elem.text and stock_code_elem.text.strip()
                    else ""
                )
                modify_date = (
                    modify_date_elem.text.strip() if modify_date_elem.text else ""
                )

                # Skip if corp_code or corp_name is empty
                if not corp_code or not corp_name:
                    continue

                company_list.append(
                    {
                        "corp_code": corp_code,
                        "corp_name": corp_name,
                        "stock_code": stock_code,
                        "modify_date": modify_date,
                    }
                )

            logger.info(f"Parsed {len(company_list)} companies from XML")
            return company_list

        except Exception as e:
            logger.error(f"Failed to parse XML file {file_path}: {e}")
            raise

    async def _find_company_by_name_async(
        self, name_keyword: str, exact_match: bool = False, limit: int = 10
    ) -> List[Dict[str, str]]:
        """
        Find companies by name keyword asynchronously.

        Args:
            name_keyword: Keyword to search in company name
            exact_match: If True, search for exact match. If False, search for partial match
            limit: Maximum number of results to return

        Returns:
            List of company dictionaries matching the search criteria
        """

        def _search_db():
            try:
                with sqlite3.connect(self.db_path) as conn:
                    cursor = conn.cursor()

                    if exact_match:
                        # Exact match search
                        cursor.execute(
                            """
                            SELECT corp_code, corp_name, stock_code, modify_date
                            FROM corporations 
                            WHERE corp_name = ?
                            ORDER BY corp_name
                            LIMIT ?
                        """,
                            (name_keyword, limit),
                        )
                    else:
                        # Partial match search using LIKE
                        search_pattern = f"%{name_keyword}%"
                        cursor.execute(
                            """
                            SELECT corp_code, corp_name, stock_code, modify_date
                            FROM corporations 
                            WHERE corp_name LIKE ?
                            ORDER BY 
                                CASE WHEN corp_name = ? THEN 1 ELSE 2 END,
                                LENGTH(corp_name),
                                corp_name
                            LIMIT ?
                        """,
                            (search_pattern, name_keyword, limit),
                        )

                    results = cursor.fetchall()

                    return [
                        {
                            "corp_code": row[0],
                            "corp_name": row[1],
                            "stock_code": row[2],
                            "modify_date": row[3],
                        }
                        for row in results
                    ]

            except Exception as e:
                logger.error(
                    f"Failed to search companies by name '{name_keyword}': {e}"
                )
                return []

        # Run database operation in thread pool to avoid blocking
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(self.executor, _search_db)

    async def _get_company_by_corp_code_async(
        self, corp_code: str
    ) -> Optional[Dict[str, str]]:
        """
        Get company information by corp_code asynchronously.

        Args:
            corp_code: DART corporation code

        Returns:
            Company dictionary if found, None otherwise
        """

        def _get_by_code():
            try:
                with sqlite3.connect(self.db_path) as conn:
                    cursor = conn.cursor()
                    cursor.execute(
                        """
                        SELECT corp_code, corp_name, stock_code, modify_date
                        FROM corporations 
                        WHERE corp_code = ?
                    """,
                        (corp_code,),
                    )

                    result = cursor.fetchone()
                    if result:
                        return {
                            "corp_code": result[0],
                            "corp_name": result[1],
                            "stock_code": result[2],
                            "modify_date": result[3],
                        }
                    return None

            except Exception as e:
                logger.error(f"Failed to get company by corp_code '{corp_code}': {e}")
                return None

        # Run database operation in thread pool to avoid blocking
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(self.executor, _get_by_code)

    @kernel_function(
        name="find_corp_code_by_name",
        description="Find corporation code by company name. Returns the best matching corp_code for DART API usage.",
    )
    async def find_corp_code_by_name(
        self,
        corp_name: Annotated[str, "Company name to search for"],
        exact_match: Annotated[
            bool, "Whether to search for exact match or partial match"
        ] = False,
    ) -> str:
        """
        Find corporation code by company name for semantic kernel usage.

        Args:
            corp_name: Company name to search for
            exact_match: If True, search for exact match. If False, search for partial match

        Returns:
            JSON string containing search results with corp_code, corp_name, stock_code
        """
        try:
            logger.info(
                f"Searching for company: '{corp_name}', exact_match: {exact_match}"
            )

            # Search for companies
            companies = await self._find_company_by_name_async(
                corp_name, exact_match=exact_match, limit=5
            )

            if not companies:
                return f"No companies found matching '{corp_name}'"

            # Format results
            result_lines = []
            result_lines.append(
                f"Found {len(companies)} company(ies) matching '{corp_name}':"
            )

            for i, company in enumerate(companies, 1):
                stock_info = (
                    f" (Stock: {company['stock_code']})"
                    if company["stock_code"]
                    else ""
                )
                result_lines.append(
                    f"{i}. {company['corp_name']}{stock_info} - Corp Code: {company['corp_code']}"
                )

            # Return the first (best) match corp_code for DART usage
            best_match = companies[0]
            result_lines.append("")
            result_lines.append(
                f"Best match corp_code for DART API: {best_match['corp_code']}"
            )

            return "\n".join(result_lines)

        except Exception as e:
            error_msg = f"Error searching for corp_code by name '{corp_name}': {str(e)}"
            logger.error(error_msg)
            return error_msg

    @kernel_function(
        name="get_company_info",
        description="Get detailed company information by corporation code",
    )
    async def get_company_info(
        self, corp_code: Annotated[str, "DART corporation code"]
    ) -> str:
        """
        Get company information by corp_code for semantic kernel usage.

        Args:
            corp_code: DART corporation code

        Returns:
            String containing company information
        """
        try:
            logger.info(f"Getting company info for corp_code: '{corp_code}'")

            company = await self._get_company_by_corp_code_async(corp_code)

            if not company:
                return f"No company found with corp_code '{corp_code}'"

            stock_info = (
                f"\nStock Code: {company['stock_code']}"
                if company["stock_code"]
                else "\nStock Code: Not Listed"
            )

            return f"""Company Information:
Corp Code: {company['corp_code']}
Company Name: {company['corp_name']}{stock_info}
Last Modified: {company['modify_date']}"""

        except Exception as e:
            error_msg = (
                f"Error getting company info for corp_code '{corp_code}': {str(e)}"
            )
            logger.error(error_msg)
            return error_msg

    async def cleanup(self):
        """Cleanup resources."""
        try:
            self.executor.shutdown(wait=True)
            logger.info("CORPPlugin cleanup completed")
        except Exception as e:
            logger.error(f"Error during CORPPlugin cleanup: {e}")


# Utility functions for direct usage (not part of semantic kernel)
def parse_corp_code_xml(file_path: str) -> List[Dict[str, str]]:
    """
    Parse CORPCODE.xml file and return list of company information.
    This is a standalone utility function for direct usage.

    Args:
        file_path: Path to CORPCODE.xml file

    Returns:
        List of company dictionaries
    """
    plugin = CORPPlugin()
    return plugin._parse_corp_code_xml(file_path)


def find_company_by_name(
    company_list: List[Dict[str, str]], name_keyword: str
) -> List[Dict[str, str]]:
    """
    Find companies by name keyword from a list of companies.
    This is a standalone utility function for direct usage.

    Args:
        company_list: List of company dictionaries
        name_keyword: Keyword to search for in company name

    Returns:
        List of matching companies
    """
    return [comp for comp in company_list if name_keyword in comp["corp_name"]]


# Example usage for testing
if __name__ == "__main__":
    import asyncio

    async def test_corp_plugin():
        """Test the CORPPlugin functionality."""
        # Initialize plugin
        plugin = CORPPlugin(xml_path="CORPCODE.xml")

        # Test company search
        result = await plugin.find_corp_code_by_name("삼성전자", exact_match=False)
        print("Search Result:")
        print(result)
        print()

        # Test company info lookup
        # Assuming we know Samsung Electronics corp_code is "00126380"
        info_result = await plugin.get_company_info("00126380")
        print("Company Info:")
        print(info_result)

        # Cleanup
        await plugin.cleanup()

    # Run test
    asyncio.run(test_corp_plugin())
