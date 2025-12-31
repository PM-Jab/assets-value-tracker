FROM postgres:17.2

# Install build dependencies and pg_cron
RUN apt-get update && apt-get install -y \
    postgresql-server-dev-17 \
    build-essential \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install pg_cron from source
RUN cd /tmp \
    && git clone https://github.com/citusdata/pg_cron.git \
    && cd pg_cron \
    && make \
    && make install