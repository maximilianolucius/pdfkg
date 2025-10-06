tar -czf /mnt/BackUps/vs/pdfkg_backup_$(date +%Y-%m-%d_%H-%M-%S).tar.gz -C /home/maxim/PycharmProjects --exclude='.venv' --exclude='venv' pdfkg &
