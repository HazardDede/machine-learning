"""Provides utility methods for I/O operations."""

import urllib.request

from tqdm import tqdm


class _DownloadProgressBar(tqdm):  # pylint: disable=too-few-public-methods
    """Helper for file download progress bar."""
    def update_to(self, b=1, bsize=1, tsize=None):  # pylint: disable=invalid-name
        """Hook for urllib to report progress."""
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)


def download_url(url, output_path):
    """Download an url to the specified output path."""
    pbar = _DownloadProgressBar(
        unit='B', unit_scale=True, miniters=1, desc=url.split('/')[-1]
    )
    with pbar:
        urllib.request.urlretrieve(url, filename=output_path, reporthook=pbar.update_to)
