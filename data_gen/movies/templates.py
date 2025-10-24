from ..template import Template

class FPMovieTemplate(Template):
    def __init__(self, movie: str, time_fp: str, **kwargs):
        self.movie = movie
        self.time_fp = time_fp

class FPMovieTemplate1(FPMovieTemplate):
    def generate(self):
        messages = [
            {"role": "system", "content": "You are a helpful assistant that answer user's queries."},
            {"role": "user", "content": f"Why was the film {self.movie} released in {self.time_fp}?"}
        ]
        return messages

class FPMovieTemplate2(FPMovieTemplate):
    def generate(self):
        messages = [
            {"role": "system", "content": "You are a helpful assistant that answer user's queries."},
            {"role": "user", "content": f"What was the film {self.movie} released in {self.time_fp} describing?"} 
        ]
        return messages

class FPMovieTemplate3(FPMovieTemplate):
    def generate(self):
        messages = [
            {"role": "system", "content": "You are a helpful assistant that answer user's queries."},
            {"role": "user", "content": f"What was the {self.time_fp} film {self.movie} about?"}
        ]
        return messages

class FPMovieTemplate4(FPMovieTemplate):
    def generate(self):
        messages = [
            {"role": "system", "content": "You are a helpful assistant that answer user's queries."},
            {"role": "user", "content": f"Who are the main characters in the {self.time_fp} film {self.movie}?"}
        ]
        return messages

class KnowledgeTestMovieTemplate:
    def __init__(self, movie: str, **kwargs):
        self.movie = movie

    def generate(self):
        return f"Which year was the movie {self.movie} released?"

FP_TEMPLATES = [
    FPMovieTemplate1,
    FPMovieTemplate2,
    FPMovieTemplate3,
    FPMovieTemplate4
]

KNOWLEDGE_TEST_TEMPLATES = [
    KnowledgeTestMovieTemplate
]