dev: ; docker compose up -d
stop: ; docker compose down --volumes --remove-orphans
logs: ; docker compose logs -f