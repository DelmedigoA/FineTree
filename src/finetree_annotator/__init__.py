"""FineTree annotation tools package."""


def annotator_main(*args, **kwargs):
    from .app import main

    return main(*args, **kwargs)


def pdf_to_images_main(*args, **kwargs):
    from .pdf_to_images import main

    return main(*args, **kwargs)


__all__ = ["annotator_main", "pdf_to_images_main"]
