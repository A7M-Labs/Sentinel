# Troubleshooting Guide 🔧

## Common Issues

### Video Processing Errors
- **Issue**: Processing timeout
  ```
  Solution: Increase timeout value in config.yaml
  ```

- **Issue**: High memory usage
  ```
  Solution: Reduce batch size in processing settings
  ```

### API Connection Issues
1. Check API key validity
2. Verify network connectivity
3. Ensure correct endpoint URLs

## Performance Optimization
- Enable caching for frequent queries
- Optimize video resolution
- Use appropriate batch sizes

## Debug Mode
Enable debug logging:
```bash 
streamlit run main.py --server.port 5001
```