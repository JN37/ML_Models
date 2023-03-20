import json

import requests


class JobSearcher:
    def __init__(self, titles, page=None, offset=0, search_limit=None):
        self.offset = offset
        self.titles = titles
        self.page = page if page is not None else 10
        self.limit = search_limit if search_limit is not None else 100

    def search_jobs(self):
        url = "https://jobsearch.api.jobtechdev.se/search"

        all_jobs = []

        while True:
            params = {
                "q": self.titles,
                "offset": self.offset,
                "limit": self.limit
            }

            response = requests.get(url, params=params)

            if response.status_code == 200:
                result = response.json()
                all_jobs.extend(result["hits"])

                if len(result["hits"]) < self.limit:
                    break
                else:
                    self.offset += self.limit
            else:
                raise Exception(f"Error {response.status_code}: {response.text}")

        return all_jobs

    def save_results(self, jobs, filename):
        job_data = []

        for job in jobs:
            job_info = {
                "Job Title": job["headline"],
                "Company": job["employer"]["name"],
                "Location": job["workplace_address"]["municipality"],
                "Publication Date": job["publication_date"],
                "Job URL": job["webpage_url"],
                "About the Job": job["description"]["text"]
            }
            job_data.append(job_info)

        with open(filename, "w", encoding="utf-8") as file:
            json.dump(job_data, file, ensure_ascii=False, indent=4)