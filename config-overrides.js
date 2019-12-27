const path = require('path')

module.exports = function override(config, env) {
  config.module.rules.push({
    test: /\.wasm$/i,
    type: 'javascript/auto',
    use: [{ loader: 'file-loader' }]
  })

  return config
}
