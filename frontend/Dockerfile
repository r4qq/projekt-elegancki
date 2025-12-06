FROM nginx:alpine

# Remove default nginx static assets
RUN rm -rf /usr/share/nginx/html/*

# Copy static assets from builder stage
COPY . /usr/share/nginx/html

# Copy custom nginx config
COPY nginx.conf /etc/nginx/conf.d/default.conf

# Grant read permissions to everyone. 
RUN chmod -R 755 /usr/share/nginx/html


EXPOSE 80

CMD ["nginx", "-g", "daemon off;"]