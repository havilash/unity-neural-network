using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using UnityEngine;
using UnityEngine.Tilemaps;

namespace Assets.Scripts.DataHandling
{
    public static class DataLoader
    {
        public static List<Dictionary<string, string>> ReadCSV(string path)
        {
            var data = new List<Dictionary<string, string>>();
            var lines = File.ReadAllLines(path);

            if (lines.Length < 2) return data;

            var headers = lines[0].Split(',');

            for (var i = 1; i < lines.Length; i++)
            {
                var values = lines[i].Split(',');
                var entry = new Dictionary<string, string>();
                for (var j = 0; j < headers.Length && j < values.Length; j++)
                {
                    entry[headers[j]] = values[j];
                }
                data.Add(entry);
            }

            return data;
        }
    }

}
