const _ = require('lodash')
const fs = require('fs')
const resolve = require('resolve-dir')


// generic experimentId matcher. index 2: experimentId, 3 or 4: experimentName
const expIdRegex = /(\-e\s+)?(([a-zA-Z0-9_]+)\-\d{4}_\d{2}_\d{2}_\d{6}|([a-zA-Z0-9_]+))/
const historyPath = './config/history.json'
const finishMsg = `
===========================================
Experiments complete. Press Ctrl+C to exit.
===========================================
`


module.exports = function(grunt) {
  process.env.NODE_ENV = grunt.option('prod') ? 'production' : 'development'

  const config = require('config')
  const dataSrc = 'data'
  const dataDest = resolve(config.data_sync_destination)
  const experiments = config.experiments
  const experimentTasks = _.map(experiments, function(name) {
    return `shell:experiment:${name}`
  })

  function writeHistory(history) {
    grunt.log.ok(`Writing updated lab history ${JSON.stringify(history, null, 2)}`)
    fs.writeFileSync(historyPath, JSON.stringify(history, null, 2))
    return history
  }

  function readHistory() {
    if (grunt.option('resume')) {
      try {
        return JSON.parse(fs.readFileSync(historyPath, 'utf8'))
      } catch (err) {
        grunt.log.ok(`No existing ${historyPath} to resume, creating new`)
        return writeHistory({})
      }
    } else {
      return {}
    }
  }

  let history = readHistory()

  function getExpId(filepath) {
    if (!fs.lstatSync(filepath).isFile()) {
      // write history on folder being created
      return filepath
    } else if (_.endsWith(filepath, '.json')) {
      // write history on json written (fallback guard)
      let expIdPath = _.join(_.initial(filepath.split('_')), '_')
      return expIdPath.split('/').pop()
    } else {
      return false
    }
  }

  function updateHistory(filepath) {
    let expId = getExpId(filepath)
    if (!expId) {
      return
    }
    const matchedPath = expId.split('/').pop().match(expIdRegex)
    if (matchedPath) {
      const experimentId = matchedPath[2]
      const experimentName = matchedPath[3] || matchedPath[4]
      history[experimentName] = experimentId
      writeHistory(history)
    }
  }

  function remoteCmd() {
    return grunt.option('remote') ? 'xvfb-run -a -s "-screen 0 1400x900x24" --' : ''
  }

  function analyzeCmd() {
    return grunt.option('analyze') ? ' -a' : ''
  }

  function bestCmd() {
    return grunt.option('best') ? '' : ' -bp'
  }

  function quietCmd() {
    return grunt.option('quiet') ? ' -q' : ''
  }

  function notiCmd(experiment) {
    return (grunt.option('prod') && !grunt.option('analyze')) ? `NOTI_SLACK_DEST='${config.NOTI_SLACK_DEST}' NOTI_SLACK_TOK='${config.NOTI_SLACK_TOK}' noti -k -t 'Experiment completed' -m '[${new Date().toISOString()}] ${experiment} on ${process.env.USER}'` : ''
  }

  function resumeExperimentStr(eStr) {
    const matchedExp = eStr.match(expIdRegex)
    if (matchedExp) {
      const experimentIdOrName = matchedExp[2]
      const experimentName = matchedExp[3] || matchedExp[4]
      if (history[experimentName]) {
        return eStr.replace(experimentIdOrName, history[experimentName])
      }
    }
    return eStr
  }

  function composeCommand(experimentStr) {
    var eStr = experimentStr
    if (grunt.option('resume')) {
      eStr = resumeExperimentStr(eStr)
    }

    // override with custom command if has 'python'
    var pyCmd = _.includes(eStr, 'python') ? eStr : `python3 main.py${analyzeCmd()}${bestCmd()}${quietCmd()} -t 5 -e ${eStr}`
    const cmd = `${remoteCmd()} ${pyCmd} | tee ./data/terminal.log; ${notiCmd(eStr)}`
    grunt.log.ok(`Composed command: ${cmd}`)
    return cmd
  }


  require('load-grunt-tasks')(grunt)

  grunt.initConfig({
    sync: {
      main: {
        files: [{
          cwd: dataSrc,
          src: ['**'],
          dest: dataDest,
        }],
        pretend: !grunt.option('prod'), // Don't do real IO
      }
    },

    watch: {
      data: {
        files: `${dataSrc}/**`,
        tasks: ['sync'],
        options: {
          debounceDelay: 20 * 60 * 1000,
          interval: 60000,
        },
      }
    },

    shell: {
      options: {
        execOptions: {
          killSignal: 'SIGINT',
          env: process.env
        }
      },
      experiment: {
        command(experimentStr) {
          return composeCommand(experimentStr)
        },
        options: {
          stdout: true
        }
      },
      finish: `echo "${finishMsg}"`,
      clear: 'rm -rf .cache __pycache__ */__pycache__ *egg-info htmlcov .coverage *.xml data/**/ data/*.log config/history.json',
    },

    concurrent: {
      default: ['watch', ['lab', 'shell:finish']],
      options: {
        logConcurrentOutput: true
      }
    },
  })

  grunt.event.on('watch', function(action, filepath, target) {
    updateHistory(filepath)
  })

  grunt.registerTask('lab', 'run all the experiments', experimentTasks)
  grunt.registerTask('lab_sync', 'run lab with auto file syncing', ['concurrent:default'])
  grunt.registerTask('default', ['lab_sync'])

  grunt.registerTask('analyze', function() {
    grunt.option('analyze', true)
    grunt.option('resume', true)
    grunt.task.run('default')
  })
  grunt.registerTask('clear', ['shell:clear'])
}
